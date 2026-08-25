#ifndef NTA_SERVING_PTHREAD_CLOCK_COMPAT_H
#define NTA_SERVING_PTHREAD_CLOCK_COMPAT_H

// CUDA 12.9's host headers do not yet match the glibc 2.43 feature-macro
// surface.  When nvcc is run without _GNU_SOURCE, libstdc++ still references
// these POSIX clock-aware pthread entry points, so provide their declarations
// explicitly.  This header contains declarations only; libc supplies the
// implementations and the serving process still links against normal pthread
// APIs.
#include <pthread.h>
#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif

int pthread_cond_clockwait(pthread_cond_t *__restrict cond,
                           pthread_mutex_t *__restrict mutex,
                           clockid_t clock_id,
                           const struct timespec *__restrict abstime);
int pthread_mutex_clocklock(pthread_mutex_t *__restrict mutex,
                            clockid_t clock_id,
                            const struct timespec *__restrict abstime);
int pthread_rwlock_clockrdlock(pthread_rwlock_t *__restrict rwlock,
                               clockid_t clock_id,
                               const struct timespec *__restrict abstime);
int pthread_rwlock_clockwrlock(pthread_rwlock_t *__restrict rwlock,
                               clockid_t clock_id,
                               const struct timespec *__restrict abstime);

#ifdef __cplusplus
}
#endif

#endif  // NTA_SERVING_PTHREAD_CLOCK_COMPAT_H
