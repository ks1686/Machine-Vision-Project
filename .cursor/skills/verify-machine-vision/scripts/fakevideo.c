/* Process-local V4L2 capture device for verification.
 * OpenCV opens /dev/videoN. When this machine has no camera, the harness
 * loads this library and serves synthetic frames to that open.
 * It is not a kernel driver and it does not touch other paths.
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <linux/version.h>
#include <linux/videodev2.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/select.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

#define MAX_FAKES 8
#define MAX_BUFS 8

struct buffer {
    void *addr;
    size_t length;
    off_t offset;
    int queued;
};

struct device {
    int fd;
    int live;
    int width;
    int height;
    int nbufs;
    int streaming;
    unsigned sequence;
    struct buffer bufs[MAX_BUFS];
    uint8_t *frame;
    size_t frame_bytes;
    char ppm_path[512];
    time_t ppm_mtime;
};

static struct device g_devs[MAX_FAKES];
static pthread_mutex_t g_mu = PTHREAD_MUTEX_INITIALIZER;
static int (*real_close)(int);
static int (*real_ioctl)(int, unsigned long, void *);
static void *(*real_mmap)(void *, size_t, int, int, int, off_t);
static int (*real_select)(int, fd_set *, fd_set *, fd_set *, struct timeval *);

static int debug_on(void) {
    const char *v = getenv("FAKEVIDEO_DEBUG");
    return v != NULL && v[0] != '\0' && v[0] != '0';
}

static void ensure_libc(void) {
    if (real_close != NULL) {
        return;
    }
    real_close = dlsym(RTLD_NEXT, "close");
    real_ioctl = dlsym(RTLD_NEXT, "ioctl");
    real_mmap = dlsym(RTLD_NEXT, "mmap");
    real_select = dlsym(RTLD_NEXT, "select");
}

__attribute__((constructor)) static void fakevideo_init(void) { ensure_libc(); }

static const char *device_path(void) {
    const char *p = getenv("FAKEVIDEO_DEVICE");
    if (p != NULL && p[0] != '\0') {
        return p;
    }
    return "/dev/video0";
}

static int is_target_path(const char *path) {
    return path != NULL && strcmp(path, device_path()) == 0;
}

static struct device *device_by_fd(int fd) {
    int i;
    for (i = 0; i < MAX_FAKES; i++) {
        if (g_devs[i].live && g_devs[i].fd == fd) {
            return &g_devs[i];
        }
    }
    return NULL;
}

/* libc syscall() already stores errno and returns -1. Do not negate rc again. */
static int sys_openat(int dirfd, const char *path, int flags, mode_t mode) {
    long rc = syscall(SYS_openat, dirfd, path, flags, (unsigned long)mode);
    if (rc < 0) {
        return -1;
    }
    return (int)rc;
}

static void clear_buffers(struct device *dev) {
    int i;
    dev->nbufs = 0;
    dev->streaming = 0;
    for (i = 0; i < MAX_BUFS; i++) {
        dev->bufs[i].addr = NULL;
        dev->bufs[i].length = 0;
        dev->bufs[i].offset = 0;
        dev->bufs[i].queued = 0;
    }
}

static int make_fake_fd(void) {
    struct device *slot = NULL;
    int fd;
    int i;
    pthread_mutex_lock(&g_mu);
    for (i = 0; i < MAX_FAKES; i++) {
        if (!g_devs[i].live) {
            slot = &g_devs[i];
            break;
        }
    }
    if (slot == NULL) {
        pthread_mutex_unlock(&g_mu);
        errno = EMFILE;
        return -1;
    }
    fd = (int)syscall(SYS_memfd_create, "fakevideo", 0);
    if (fd < 0) {
        pthread_mutex_unlock(&g_mu);
        return -1;
    }
    memset(slot, 0, sizeof(*slot));
    slot->fd = fd;
    slot->live = 1;
    slot->width = 640;
    slot->height = 480;
    pthread_mutex_unlock(&g_mu);
    return fd;
}

static int open_maybe(const char *path, int flags, mode_t mode, int dirfd, int have_dir) {
    if (is_target_path(path)) {
        return make_fake_fd();
    }
    if (have_dir) {
        return sys_openat(dirfd, path, flags, mode);
    }
    return sys_openat(AT_FDCWD, path, flags, mode);
}

static mode_t mode_from_flags(int flags, va_list ap) {
    if (flags & (O_CREAT | O_TMPFILE)) {
        return (mode_t)va_arg(ap, int);
    }
    return 0;
}

int open(const char *pathname, int flags, ...) {
    mode_t mode;
    va_list ap;
    va_start(ap, flags);
    mode = mode_from_flags(flags, ap);
    va_end(ap);
    return open_maybe(pathname, flags, mode, AT_FDCWD, 0);
}

int open64(const char *pathname, int flags, ...) {
    mode_t mode;
    va_list ap;
    va_start(ap, flags);
    mode = mode_from_flags(flags, ap);
    va_end(ap);
    return open_maybe(pathname, flags, mode, AT_FDCWD, 0);
}

int openat(int dirfd, const char *pathname, int flags, ...) {
    mode_t mode;
    va_list ap;
    va_start(ap, flags);
    mode = mode_from_flags(flags, ap);
    va_end(ap);
    if (pathname[0] == '/') {
        return open_maybe(pathname, flags, mode, dirfd, 1);
    }
    return sys_openat(dirfd, pathname, flags, mode);
}

int openat64(int dirfd, const char *pathname, int flags, ...) {
    mode_t mode;
    va_list ap;
    va_start(ap, flags);
    mode = mode_from_flags(flags, ap);
    va_end(ap);
    if (pathname[0] == '/') {
        return open_maybe(pathname, flags, mode, dirfd, 1);
    }
    return sys_openat(dirfd, pathname, flags, mode);
}

int close(int fd) {
    struct device *dev;
    ensure_libc();
    pthread_mutex_lock(&g_mu);
    dev = device_by_fd(fd);
    if (dev != NULL) {
        free(dev->frame);
        dev->frame = NULL;
        dev->live = 0;
    }
    pthread_mutex_unlock(&g_mu);
    return real_close(fd);
}

static int load_ppm(const char *path, uint8_t **rgb, int *w, int *h) {
    FILE *fp;
    char magic[3];
    int width = 0;
    int height = 0;
    int maxval = 0;
    int c;
    size_t n;
    uint8_t *data;
    fp = fopen(path, "rb");
    if (fp == NULL) {
        return -1;
    }
    if (fread(magic, 1, 3, fp) != 3 || magic[0] != 'P' || magic[1] != '6' ||
        (magic[2] != '\n' && magic[2] != ' ' && magic[2] != '\t' && magic[2] != '\r')) {
        fclose(fp);
        return -1;
    }
    if (magic[2] != '\n') {
        ungetc(magic[2], fp);
    }
    while ((c = fgetc(fp)) == '#') {
        while ((c = fgetc(fp)) != '\n' && c != EOF) {
        }
    }
    if (c == EOF) {
        fclose(fp);
        return -1;
    }
    ungetc(c, fp);
    if (fscanf(fp, "%d %d", &width, &height) != 2 || width <= 0 || height <= 0 ||
        width > 4096 || height > 4096) {
        fclose(fp);
        return -1;
    }
    while ((c = fgetc(fp)) == '#') {
        while ((c = fgetc(fp)) != '\n' && c != EOF) {
        }
    }
    if (c == EOF) {
        fclose(fp);
        return -1;
    }
    ungetc(c, fp);
    if (fscanf(fp, "%d", &maxval) != 1 || maxval != 255) {
        fclose(fp);
        return -1;
    }
    c = fgetc(fp);
    if (c != '\n') {
        fclose(fp);
        return -1;
    }
    n = (size_t)width * (size_t)height * 3u;
    data = malloc(n);
    if (data == NULL) {
        fclose(fp);
        return -1;
    }
    if (fread(data, 1, n, fp) != n) {
        free(data);
        fclose(fp);
        return -1;
    }
    fclose(fp);
    *rgb = data;
    *w = width;
    *h = height;
    return 0;
}

static void fill_gray(struct device *dev, uint8_t value) {
    size_t i;
    size_t n = (size_t)dev->width * (size_t)dev->height * 3u;
    if (dev->frame == NULL || dev->frame_bytes != n) {
        free(dev->frame);
        dev->frame = malloc(n);
        dev->frame_bytes = dev->frame != NULL ? n : 0;
    }
    if (dev->frame == NULL) {
        return;
    }
    for (i = 0; i < n; i++) {
        dev->frame[i] = value;
    }
}

static void refresh_frame(struct device *dev) {
    const char *path = getenv("FAKEVIDEO_PPM");
    struct stat st;
    uint8_t *rgb = NULL;
    int sw = 0;
    int sh = 0;
    size_t n;
    int y;
    int x;
    if (path == NULL || path[0] == '\0') {
        fill_gray(dev, 40);
        dev->ppm_path[0] = '\0';
        return;
    }
    if (stat(path, &st) != 0) {
        fill_gray(dev, 40);
        return;
    }
    n = (size_t)dev->width * (size_t)dev->height * 3u;
    if (dev->frame != NULL && dev->frame_bytes == n && strcmp(dev->ppm_path, path) == 0 &&
        dev->ppm_mtime == st.st_mtime) {
        return;
    }
    if (load_ppm(path, &rgb, &sw, &sh) != 0) {
        fill_gray(dev, 40);
        return;
    }
    free(dev->frame);
    dev->frame = malloc(n);
    if (dev->frame == NULL) {
        free(rgb);
        dev->frame_bytes = 0;
        return;
    }
    dev->frame_bytes = n;
    for (y = 0; y < dev->height; y++) {
        int sy = (int)(((long)y * sh) / dev->height);
        for (x = 0; x < dev->width; x++) {
            int sx = (int)(((long)x * sw) / dev->width);
            size_t si = ((size_t)sy * (size_t)sw + (size_t)sx) * 3u;
            size_t di = ((size_t)y * (size_t)dev->width + (size_t)x) * 3u;
            dev->frame[di + 0] = rgb[si + 0];
            dev->frame[di + 1] = rgb[si + 1];
            dev->frame[di + 2] = rgb[si + 2];
        }
    }
    free(rgb);
    snprintf(dev->ppm_path, sizeof(dev->ppm_path), "%s", path);
    dev->ppm_mtime = st.st_mtime;
}

static void set_pix(struct v4l2_pix_format *pix, int width, int height) {
    memset(pix, 0, sizeof(*pix));
    pix->width = (__u32)width;
    pix->height = (__u32)height;
    pix->pixelformat = V4L2_PIX_FMT_RGB24;
    pix->field = V4L2_FIELD_NONE;
    pix->bytesperline = (__u32)width * 3u;
    pix->sizeimage = pix->bytesperline * (__u32)height;
    pix->colorspace = V4L2_COLORSPACE_SRGB;
}

static int on_ioctl(struct device *dev, unsigned long request, void *arg) {
    unsigned nr = _IOC_NR(request);
    if (debug_on()) {
        fprintf(stderr, "fakevideo: ioctl nr=%u size=%lu\n", nr, (unsigned long)_IOC_SIZE(request));
    }
    if (arg == NULL && nr != 18) {
        errno = EFAULT;
        return -1;
    }
    switch (nr) {
    case _IOC_NR(VIDIOC_QUERYCAP): {
        struct v4l2_capability *cap = arg;
        memset(cap, 0, sizeof(*cap));
        snprintf((char *)cap->driver, sizeof(cap->driver), "fakevideo");
        snprintf((char *)cap->card, sizeof(cap->card), "verify-machine-vision");
        snprintf((char *)cap->bus_info, sizeof(cap->bus_info), "platform:fakevideo");
        cap->version = KERNEL_VERSION(6, 12, 0);
        cap->capabilities = V4L2_CAP_VIDEO_CAPTURE | V4L2_CAP_STREAMING;
        return 0;
    }
    case _IOC_NR(VIDIOC_G_INPUT): {
        int *channel = arg;
        *channel = 0;
        return 0;
    }
    case _IOC_NR(VIDIOC_S_INPUT):
        return 0;
    case _IOC_NR(VIDIOC_ENUMINPUT): {
        struct v4l2_input *input = arg;
        if (input->index != 0) {
            errno = EINVAL;
            return -1;
        }
        memset(input, 0, sizeof(*input));
        snprintf((char *)input->name, sizeof(input->name), "synthetic");
        input->type = V4L2_INPUT_TYPE_CAMERA;
        return 0;
    }
    case _IOC_NR(VIDIOC_G_FMT): {
        struct v4l2_format *fmt = arg;
        if (fmt->type != V4L2_BUF_TYPE_VIDEO_CAPTURE) {
            errno = EINVAL;
            return -1;
        }
        set_pix(&fmt->fmt.pix, dev->width, dev->height);
        fmt->type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        return 0;
    }
    case _IOC_NR(VIDIOC_S_FMT): {
        struct v4l2_format *fmt = arg;
        /* OpenCV 4.11 copies BGR24 as a 1-row buffer. RGB24 is converted
         * with the real width and height, so that is the format we offer. */
        if (fmt->type != V4L2_BUF_TYPE_VIDEO_CAPTURE ||
            fmt->fmt.pix.pixelformat != V4L2_PIX_FMT_RGB24) {
            errno = EINVAL;
            return -1;
        }
        if (fmt->fmt.pix.width > 0 && fmt->fmt.pix.width <= 4096) {
            dev->width = (int)fmt->fmt.pix.width;
        }
        if (fmt->fmt.pix.height > 0 && fmt->fmt.pix.height <= 4096) {
            dev->height = (int)fmt->fmt.pix.height;
        }
        set_pix(&fmt->fmt.pix, dev->width, dev->height);
        fmt->type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        dev->ppm_mtime = 0;
        return 0;
    }
    case _IOC_NR(VIDIOC_S_PARM):
    case _IOC_NR(VIDIOC_G_PARM): {
        struct v4l2_streamparm *sp = arg;
        sp->type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        memset(&sp->parm, 0, sizeof(sp->parm));
        sp->parm.capture.capability = V4L2_CAP_TIMEPERFRAME;
        sp->parm.capture.timeperframe.numerator = 1;
        sp->parm.capture.timeperframe.denominator = 30;
        return 0;
    }
    case _IOC_NR(VIDIOC_REQBUFS): {
        struct v4l2_requestbuffers *req = arg;
        if (req->type != V4L2_BUF_TYPE_VIDEO_CAPTURE || req->memory != V4L2_MEMORY_MMAP) {
            errno = EINVAL;
            return -1;
        }
        clear_buffers(dev);
        if (req->count > MAX_BUFS) {
            req->count = MAX_BUFS;
        }
        dev->nbufs = (int)req->count;
        return 0;
    }
    case _IOC_NR(VIDIOC_QUERYBUF): {
        struct v4l2_buffer *buf = arg;
        if (buf->type != V4L2_BUF_TYPE_VIDEO_CAPTURE || buf->memory != V4L2_MEMORY_MMAP ||
            buf->index >= (unsigned)dev->nbufs) {
            errno = EINVAL;
            return -1;
        }
        buf->length = (unsigned)dev->width * (unsigned)dev->height * 3u;
        buf->m.offset = (__u32)buf->index * 0x01000000u;
        dev->bufs[buf->index].offset = (off_t)buf->m.offset;
        dev->bufs[buf->index].length = buf->length;
        buf->flags = V4L2_BUF_FLAG_MAPPED;
        return 0;
    }
    case _IOC_NR(VIDIOC_QBUF): {
        struct v4l2_buffer *buf = arg;
        if (buf->index >= (unsigned)dev->nbufs) {
            errno = EINVAL;
            return -1;
        }
        dev->bufs[buf->index].queued = 1;
        buf->flags = V4L2_BUF_FLAG_MAPPED | V4L2_BUF_FLAG_QUEUED;
        return 0;
    }
    case _IOC_NR(VIDIOC_DQBUF): {
        struct v4l2_buffer *buf = arg;
        int i;
        for (i = 0; i < dev->nbufs; i++) {
            if (!dev->bufs[i].queued || dev->bufs[i].addr == NULL) {
                continue;
            }
            refresh_frame(dev);
            if (dev->frame != NULL && dev->bufs[i].length >= dev->frame_bytes) {
                memcpy(dev->bufs[i].addr, dev->frame, dev->frame_bytes);
            }
            dev->bufs[i].queued = 0;
            memset(buf, 0, sizeof(*buf));
            buf->index = (__u32)i;
            buf->type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            buf->memory = V4L2_MEMORY_MMAP;
            buf->bytesused = (unsigned)dev->frame_bytes;
            buf->length = (unsigned)dev->bufs[i].length;
            buf->flags = V4L2_BUF_FLAG_MAPPED | V4L2_BUF_FLAG_DONE;
            buf->field = V4L2_FIELD_NONE;
            buf->sequence = dev->sequence++;
            gettimeofday(&buf->timestamp, NULL);
            return 0;
        }
        errno = EAGAIN;
        return -1;
    }
    case _IOC_NR(VIDIOC_STREAMON):
        dev->streaming = 1;
        return 0;
    case _IOC_NR(VIDIOC_STREAMOFF):
        dev->streaming = 0;
        return 0;
    case _IOC_NR(VIDIOC_ENUM_FMT): {
        struct v4l2_fmtdesc *desc = arg;
        if (desc->index != 0 || desc->type != V4L2_BUF_TYPE_VIDEO_CAPTURE) {
            errno = EINVAL;
            return -1;
        }
        desc->pixelformat = V4L2_PIX_FMT_RGB24;
        snprintf((char *)desc->description, sizeof(desc->description), "RGB24");
        return 0;
    }
    default:
        errno = EINVAL;
        return -1;
    }
}

int ioctl(int fd, unsigned long request, ...) {
    void *arg;
    struct device *dev;
    va_list ap;
    int rc;
    ensure_libc();
    va_start(ap, request);
    arg = va_arg(ap, void *);
    va_end(ap);
    if (_IOC_TYPE(request) != 'V') {
        return real_ioctl(fd, request, arg);
    }
    pthread_mutex_lock(&g_mu);
    dev = device_by_fd(fd);
    if (dev == NULL) {
        pthread_mutex_unlock(&g_mu);
        return real_ioctl(fd, request, arg);
    }
    rc = on_ioctl(dev, request, arg);
    pthread_mutex_unlock(&g_mu);
    return rc;
}

static void *mmap_fake(struct device *dev, size_t length, off_t offset) {
    int i;
    void *p;
    for (i = 0; i < dev->nbufs; i++) {
        if (dev->bufs[i].offset != offset) {
            continue;
        }
        p = real_mmap(NULL, length, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (p == MAP_FAILED) {
            return MAP_FAILED;
        }
        dev->bufs[i].addr = p;
        dev->bufs[i].length = length;
        return p;
    }
    errno = EINVAL;
    return MAP_FAILED;
}

void *mmap(void *addr, size_t length, int prot, int flags, int fd, off_t offset) {
    struct device *dev;
    void *p;
    ensure_libc();
    pthread_mutex_lock(&g_mu);
    dev = device_by_fd(fd);
    if (dev == NULL) {
        pthread_mutex_unlock(&g_mu);
        return real_mmap(addr, length, prot, flags, fd, offset);
    }
    p = mmap_fake(dev, length, offset);
    pthread_mutex_unlock(&g_mu);
    return p;
}

void *mmap64(void *addr, size_t length, int prot, int flags, int fd, off_t offset) {
    return mmap(addr, length, prot, flags, fd, offset);
}

int select(int nfds, fd_set *readfds, fd_set *writefds, fd_set *exceptfds, struct timeval *timeout) {
    fd_set original;
    int saw_fake = 0;
    int i;
    int rc;
    struct timeval zero;
    ensure_libc();
    if (readfds != NULL) {
        pthread_mutex_lock(&g_mu);
        for (i = 0; i < MAX_FAKES; i++) {
            if (g_devs[i].live && g_devs[i].fd < nfds && FD_ISSET(g_devs[i].fd, readfds)) {
                saw_fake = 1;
            }
        }
        pthread_mutex_unlock(&g_mu);
    }
    if (!saw_fake) {
        return real_select(nfds, readfds, writefds, exceptfds, timeout);
    }
    original = *readfds;
    zero.tv_sec = 0;
    zero.tv_usec = 0;
    rc = real_select(nfds, readfds, writefds, exceptfds, &zero);
    if (rc < 0) {
        return rc;
    }
    pthread_mutex_lock(&g_mu);
    for (i = 0; i < MAX_FAKES; i++) {
        int fd = g_devs[i].fd;
        if (!g_devs[i].live || fd >= nfds || !FD_ISSET(fd, &original)) {
            continue;
        }
        if (!FD_ISSET(fd, readfds)) {
            FD_SET(fd, readfds);
            rc++;
        }
    }
    pthread_mutex_unlock(&g_mu);
    return rc;
}
