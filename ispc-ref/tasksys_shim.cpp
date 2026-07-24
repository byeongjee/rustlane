// The ISPC task ABI implemented here (ISPCAlloc/ISPCLaunch/ISPCSync and the
// TaskFuncPtr signature) is defined by Intel's ISPC (BSD-3-Clause); the serial
// implementation below is original to this project. See THIRD-PARTY.md.
//
// Minimal serial task-runtime shim for ISPC kernels that also emit *_tasks
// (launch) entry points. The bench driver only calls the non-task entries, so
// these are here purely to resolve ISPCAlloc/ISPCLaunch/ISPCSync at link time.
// Implemented as a correct *serial* runtime (tasks run inline) rather than a
// stub, so linking one object is enough and behavior is well-defined if ever
// called. This is NOT ISPC's threaded tasksys.

#include <cstdint>
#include <cstdlib>
#include <vector>

typedef void (*TaskFuncPtr)(void *data, int threadIndex, int threadCount,
                            int taskIndex, int taskCount, int taskIndex0,
                            int taskIndex1, int taskIndex2, int taskCount0,
                            int taskCount1, int taskCount2);

struct Task {
    TaskFuncPtr func;
    void *data;
    int count0, count1, count2;
};

// One pending task list per launch group (handle points to a TaskGroup).
struct TaskGroup {
    std::vector<Task> tasks;
    std::vector<void *> allocations;
};

extern "C" void *ISPCAlloc(void **handlePtr, int64_t size, int32_t alignment) {
    TaskGroup *g = reinterpret_cast<TaskGroup *>(*handlePtr);
    if (!g) {
        g = new TaskGroup();
        *handlePtr = g;
    }
    void *mem = nullptr;
    if (posix_memalign(&mem, alignment < (int32_t)sizeof(void *) ? sizeof(void *)
                                                                 : alignment,
                       size) != 0)
        mem = nullptr;
    g->allocations.push_back(mem);
    return mem;
}

extern "C" void ISPCLaunch(void **handlePtr, void *f, void *data, int count0,
                           int count1, int count2) {
    TaskGroup *g = reinterpret_cast<TaskGroup *>(*handlePtr);
    if (!g) {
        g = new TaskGroup();
        *handlePtr = g;
    }
    Task t;
    t.func = reinterpret_cast<TaskFuncPtr>(f);
    t.data = data;
    t.count0 = count0;
    t.count1 = count1;
    t.count2 = count2;
    g->tasks.push_back(t);
}

extern "C" void ISPCSync(void *handle) {
    TaskGroup *g = reinterpret_cast<TaskGroup *>(handle);
    if (!g)
        return;
    for (const Task &t : g->tasks) {
        int total = t.count0 * t.count1 * t.count2;
        for (int i = 0; i < total; ++i) {
            int ti0 = i % t.count0;
            int ti1 = (i / t.count0) % t.count1;
            int ti2 = i / (t.count0 * t.count1);
            t.func(t.data, 0, 1, i, total, ti0, ti1, ti2, t.count0, t.count1,
                   t.count2);
        }
    }
    for (void *p : g->allocations)
        free(p);
    delete g;
}
