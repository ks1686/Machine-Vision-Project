"""Synthetic frontal face. Not a photograph of a real person."""
import numpy as np
import cv2

def _noise(h, w, rng, scale=8):
    n = rng.normal(0, scale, (h // 2, w // 2)).astype(np.float32)
    return cv2.resize(n, (w, h), interpolation=cv2.INTER_LINEAR)

def make_face(seed=1, iris=(40, 50, 90), skin=(90, 120, 170), hair=(25, 30, 40), w=1280, h=720):
    rng = np.random.default_rng(seed)
    img = np.zeros((h, w, 3), np.float32)
    # room background
    yy = np.linspace(0, 1, h)[:, None]
    xx = np.linspace(0, 1, w)[None, :]
    img[:, :, 0] = 70 + 30 * yy
    img[:, :, 1] = 78 + 20 * (1 - xx)
    img[:, :, 2] = 86 + 15 * yy
    img += _noise(h, w, rng, 4)[:, :, None]

    cx, cy = w * 0.50, h * 0.46
    face_rx, face_ry = w * 0.18, h * 0.34

    def ellipse_mask(cx, cy, rx, ry, rot=0, blur=0):
        m = np.zeros((h, w), np.uint8)
        cv2.ellipse(m, (int(cx), int(cy)), (int(rx), int(ry)), rot, 0, 360, 255, -1)
        if blur:
            m = cv2.GaussianBlur(m, (0, 0), blur)
        return m.astype(np.float32) / 255.0

    def paint(mask, color, add=False):
        c = np.array(color, np.float32)
        if add:
            img[:] += mask[:, :, None] * c
        else:
            img[:] = img * (1 - mask[:, :, None]) + c * mask[:, :, None]

    # shoulders and neck
    paint(ellipse_mask(cx, h * 0.92, w * 0.28, h * 0.22, blur=3), (skin[0] * 0.7, skin[1] * 0.7, skin[2] * 0.75))
    paint(ellipse_mask(cx, h * 0.72, w * 0.07, h * 0.16, blur=2), skin)

    # hair behind
    paint(ellipse_mask(cx, cy - face_ry * 0.15, face_rx * 1.18, face_ry * 1.22, blur=2), hair)
    # face
    face = ellipse_mask(cx, cy, face_rx, face_ry, blur=1.2)
    paint(face, skin)
    # shading: forehead light, cheek, jaw shadow
    shade = np.zeros((h, w), np.float32)
    cv2.ellipse(shade, (int(cx), int(cy - face_ry * 0.35)), (int(face_rx * 0.7), int(face_ry * 0.35)), 0, 0, 360, 1, -1)
    shade = cv2.GaussianBlur(shade, (0, 0), 18)
    img += shade[:, :, None] * np.array([18, 16, 12], np.float32)
    jaw = ellipse_mask(cx, cy + face_ry * 0.45, face_rx * 0.8, face_ry * 0.28, blur=8)
    img -= jaw[:, :, None] * np.array([16, 12, 10], np.float32)
    # side shadow
    side = ellipse_mask(cx + face_rx * 0.55, cy, face_rx * 0.35, face_ry * 0.8, blur=10)
    img -= side[:, :, None] * np.array([22, 16, 12], np.float32)

    # ears
    for sgn in (-1, 1):
        paint(ellipse_mask(cx + sgn * face_rx * 0.98, cy + face_ry * 0.05, face_rx * 0.16, face_ry * 0.22, blur=1), 
              (skin[0] * 0.85, skin[1] * 0.8, skin[2] * 0.82))

    # hair front bangs
    bangs = ellipse_mask(cx, cy - face_ry * 0.72, face_rx * 1.05, face_ry * 0.42, blur=1)
    # cut bangs above eyes
    cut = np.ones((h, w), np.float32)
    cut[int(cy - face_ry * 0.15):, :] = 0
    bangs = bangs * cut
    paint(bangs, hair)

    # eyebrows
    for sgn in (-1, 1):
        ex = cx + sgn * face_rx * 0.38
        ey = cy - face_ry * 0.18
        m = np.zeros((h, w), np.uint8)
        cv2.ellipse(m, (int(ex), int(ey)), (int(face_rx * 0.22), int(face_ry * 0.045)), sgn * -12, 200, 340, 255, -1)
        m = cv2.GaussianBlur(m, (0, 0), 1.2)
        paint(m.astype(np.float32) / 255.0, (hair[0] + 10, hair[1] + 8, hair[2] + 8))

    # eyes
    def eye(sgn):
        ex = cx + sgn * face_rx * 0.38
        ey = cy - face_ry * 0.02
        paint(ellipse_mask(ex, ey, face_rx * 0.16, face_ry * 0.07, blur=0.6), (240, 240, 245))
        paint(ellipse_mask(ex, ey, face_rx * 0.075, face_ry * 0.055, blur=0.4), iris)
        # iris rings
        ring = ellipse_mask(ex, ey, face_rx * 0.055, face_ry * 0.04, blur=0.3)
        img[:] = img * (1 - ring[:, :, None] * 0.35) + np.array(iris, np.float32) * 0.7 * ring[:, :, None]
        paint(ellipse_mask(ex, ey, face_rx * 0.035, face_ry * 0.028, blur=0.2), (15, 15, 18))
        paint(ellipse_mask(ex - face_rx * 0.02, ey - face_ry * 0.015, face_rx * 0.012, face_ry * 0.01, blur=0.2), (250, 250, 255))
        # upper lid
        lid = np.zeros((h, w), np.uint8)
        cv2.ellipse(lid, (int(ex), int(ey)), (int(face_rx * 0.17), int(face_ry * 0.08)), 0, 200, 340, 255, 2)
        lid = cv2.GaussianBlur(lid, (0, 0), 0.6)
        paint(lid.astype(np.float32) / 255.0, (40, 35, 35))

    eye(-1)
    eye(1)

    # nose
    nx, ny = cx, cy + face_ry * 0.18
    bridge = ellipse_mask(nx, ny - face_ry * 0.08, face_rx * 0.05, face_ry * 0.16, blur=2)
    img += bridge[:, :, None] * np.array([10, 8, 6], np.float32)
    for sgn in (-1, 1):
        paint(ellipse_mask(nx + sgn * face_rx * 0.06, ny + face_ry * 0.08, face_rx * 0.035, face_ry * 0.025, blur=0.8),
              (skin[0] * 0.65, skin[1] * 0.6, skin[2] * 0.62))

    # mouth
    mx, my = cx, cy + face_ry * 0.42
    paint(ellipse_mask(mx, my, face_rx * 0.16, face_ry * 0.045, blur=0.8), (70, 80, 150))
    paint(ellipse_mask(mx, my - face_ry * 0.01, face_rx * 0.14, face_ry * 0.02, blur=0.5), (90, 100, 170))
    line = np.zeros((h, w), np.uint8)
    cv2.line(line, (int(mx - face_rx * 0.12), int(my)), (int(mx + face_rx * 0.12), int(my)), 255, 2)
    line = cv2.GaussianBlur(line, (0, 0), 0.7)
    img -= line[:, :, None] * np.array([20, 15, 10], np.float32)

    # pores / micro contrast so the blur gate passes
    pores = _noise(h, w, rng, 7)
    img += face[:, :, None] * pores[:, :, None] * np.array([0.6, 0.5, 0.4], np.float32)

    img = np.clip(img, 0, 255).astype(np.uint8)
    return img

def shift_nose(img, dx=30):
    """Move the center band so the estimated yaw misses the enrolled face."""
    h, w = img.shape[:2]
    x0, x1 = int(w * 0.42), int(w * 0.58)
    out = img.copy()
    band = img[:, x0:x1]
    matrix = np.float32([[1, 0, dx], [0, 1, 0]])
    moved = cv2.warpAffine(band, matrix, (band.shape[1], band.shape[0]), borderMode=cv2.BORDER_REFLECT)
    out[:, x0:x1] = moved
    return out


def write_ppm(path, bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    height, width = rgb.shape[:2]
    with open(path, "wb") as handle:
        handle.write(f"P6\n{width} {height}\n255\n".encode())
        handle.write(rgb.tobytes())


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Write a synthetic face PPM. Not a photograph.")
    parser.add_argument("--kind", choices=("frontal", "shifted"), required=True)
    parser.add_argument("--ppm", required=True)
    args = parser.parse_args()
    image = make_face(1)
    if args.kind == "shifted":
        image = shift_nose(image, 30)
    write_ppm(args.ppm, image)


if __name__ == "__main__":
    main()
