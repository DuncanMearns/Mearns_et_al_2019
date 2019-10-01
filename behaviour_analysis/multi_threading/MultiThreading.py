import threading
import Queue
import time


class NewThread(threading.Thread):
    """Class for handling individual threads in MultiThreading objects."""

    def __init__(self, thread_ID, parent):
        threading.Thread.__init__(self)
        self.thread_ID = thread_ID
        self.parent = parent

    def run(self):
        print 'Starting thread-{}'.format(self.thread_ID)
        self.parent._start_thread(self.thread_ID)
        print 'Exiting thread-{}'.format(self.thread_ID)


class MultiThreading(object):
    """Base class for performing multi-threading.

    Parameters
    ----------
    n_threads : int
        Number of threads to run in parallel.

    Notes
    -----
    The _run_on_thread method should be overwritten in child classes and should take a single argument and execute some
    code using that argument. The main thread is handled by the _run method, which takes a list of arguments and passes
    them individually to to _run_on_thread.

    Call structure:
    _run(*args) -> spawns n threads -> each thread executes _run_on_thread(arg)
    """

    def __init__(self, n_threads):
        self.exit_flag = 0
        self.queue_lock = threading.Lock()
        self.q = Queue.Queue()
        self.n_threads = n_threads
        self.threads = []

    def _spawn_new_thread(self, thread_ID):
        new_thread = NewThread(thread_ID, self)
        new_thread.start()
        self.threads.append(new_thread)

    def _start_thread(self, thread_ID):
        while not self.exit_flag:
            self.queue_lock.acquire()
            if not self.q.empty():
                arg = self.q.get()
                self.queue_lock.release()
                print 'Thread-{}: {}'.format(thread_ID, arg)
                self._run_on_thread(arg)
            else:
                self.queue_lock.release()
            time.sleep(1)

    def _run(self, *args):

        thread_list = range(self.n_threads)
        for thread_ID in thread_list:
            self._spawn_new_thread(thread_ID)

        self.queue_lock.acquire()
        for arg in args:
            self.q.put(arg)
        self.queue_lock.release()

        while not self.q.empty():
            pass

        self.exit_flag = 1

        for t in self.threads:
            t.join()
        print 'Exiting main thread.'

    def _run_on_thread(self, arg):
        pass
