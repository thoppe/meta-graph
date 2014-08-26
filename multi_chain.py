import multiprocessing, time, traceback, logging

logger = logging.getLogger()
logging.basicConfig()
#logger.setLevel(logging.DEBUG)

# Generic object that stops Consumers
class PoisonPill(object):  
    def __repr__(self): return "PoisonPill"

class Consumer(multiprocessing.Process):
    
    def __init__(self, task_queue, result_queue): 
        multiprocessing.Process.__init__(self)
        self.task_queue   = task_queue
        self.result_queue = result_queue

    def ate_poison(self, task):
        return type(task) == PoisonPill

    def run(self):
        proc_name = self.name
        while True:
            logger.debug("{} is waiting".format(proc_name))
            next_task = self.task_queue.get()
            logger.debug("{} has got {}".format(proc_name, next_task))

            if self.ate_poison(next_task):
                return

            try:
                result = next_task()
            except Exception:
                logger.critical(traceback.format_exc())
                exit()

            logger.debug( "{} responded with {}".format(proc_name, result) )

            self.result_queue.put(result)

        return

class Generic_Task(object):
    def __init__(self, func, *args, **kwargs):
        self.func = func
        self.args = args
        self.kwargs = kwargs
    def __call__(self):
        return self.func(*self.args, **self.kwargs)
    def __repr__(self):
        return "{}{}".format(self.func, self.args)


class multi_Manager(object):
    def __init__(self, source, TASK_CHAIN, SERIAL_CHAIN,
                 chunksize=50, procs=None, cycle_sleep=0):

        self.chunksize = chunksize
        self.cycle_sleep = cycle_sleep
        self.task_n = len(TASK_CHAIN)
        self.task_range = range(self.task_n)

        if procs == None:
            self.procs = multiprocessing.cpu_count()
        else:
            self.procs = procs

        self.source = source
        self.func_T = TASK_CHAIN
        self.func_S = SERIAL_CHAIN

        assert(len(TASK_CHAIN) == len(SERIAL_CHAIN))

        JQ = multiprocessing.JoinableQueue
        RQ = multiprocessing.Queue
        self.T_input  = [JQ() for _ in self.task_range]
        self.S_input  = [RQ() for _ in self.task_range]
        self.S_output = [RQ() for _ in self.task_range]

        self.all_Q = [self.T_input,self.S_input,self.S_output]

        self.C = [[Consumer(t,q) for _ in range(self.procs)] 
                  for t,q in zip(self.T_input,self.S_input)]

        # Start consumers
        for c in self.consumer_iter():
            c.start()

        self._empty_source = False
        self._is_complete  = False

    def consumer_iter(self):
        for C_block in self.C:
            for c in C_block: yield c

    def run(self):
        while not self._is_complete:
            self.cycle()
            time.sleep(self.cycle_sleep)


    def is_complete(self): 
        return self._is_complete
    
    def process_serial(self, k):
        consumers = self.C[k]
        S_in = self.S_input[k]
        S_out = self.S_output[k]
        f = self.func_S[k]

        while S_in.qsize():
            val    = S_in.get()
            if not type(val)==PoisonPill:
                result = f(val)
                S_out.put(result)

    def process_final_serial(self):
        while not self.S_output[-1].empty():
            x = self.S_output[-1].get()

    def count_free_slots(self):
        free = self.chunksize
        for T_in in self.T_input:
            free -= T_in.qsize()
        return free

    def pull_source(self):
        try:
            val = next(self.source)
            return val
        except StopIteration:
            self._empty_source   = True
            return None

    def _poison_queue(self, Q):
        for _ in range(self.procs):
            Q.put(PoisonPill())
        #Q.close()
        #Q.join_thread()       

    def get_qsize(self):
        return [[Q.qsize() for Q in QX] for QX in self.all_Q]

    def load_queue(self, k, free_tasks=10**8):
        # Step through queues backwards
        # ignore first T and last S

        consumers = self.C[k]
        S_out = self.S_output[k]
        T_in = self.T_input[k+1]
        f     = self.func_T[k+1]

        while not S_out.empty():
            val  = S_out.get()
            if type(val) != PoisonPill:
                task = Generic_Task(f, val)
                T_in.put(task)

            free_tasks -= 1
            if not free_tasks: return False

        return True

            
    def shutdown(self):

        logger.debug("Shutting down")
        logger.debug("Check status: %s" % self.get_qsize()) 

        for level in range(self.task_n):          
            Q  = self.all_Q[0][level]
            self._poison_queue(Q)
            self.process_serial(level)

            Q  = self.all_Q[1][level]
            self._poison_queue(Q)
            self.process_serial(level)

            Q  = self.all_Q[2][level]
            self._poison_queue(Q)

            if level < self.task_n-1:
                self.load_queue(level)
            else:
                self.process_final_serial()

            logger.debug("Check status: %s" % self.get_qsize()) 

        self._is_complete  = True

    
    def is_empty(self):
        ''' Check if all queues are empty, including source '''
        if not self._empty_source: return False

        logger.debug("Empty status: %s" % self.get_qsize())
        for QX in [self.T_input,self.S_input,self.S_output]:
            for Q in QX:
                if Q.qsize() > 0: return False

        return True

    def cycle(self):

        for k in range(self.task_n):
            self.process_serial(k)

        # Pulling anything off of the last serial queue and discard it
        self.process_final_serial()

        free = self.count_free_slots()

        if not free: return False

        # Step through queues backwards
        # ignore first T and last S
        for k in range(self.task_n-1):
            status = self.load_queue(k, free)
            if not status: return status

        # Pull from the source if still here
        for n in range(free):
            val = self.pull_source()
            if self._empty_source:
                #logger.debug("Source is now empty")
                break

            task = Generic_Task(self.func_T[0], val)
            self.T_input[0].put( task )

        if self.is_empty():
            self.shutdown()


# These test methods have to be global to work with 'spawn' method!

def serial_a(x):
    print ("A result: ", x)
    return x

def serial_b(x):
    print ("B result: ", x)
    return x

def mul2(x):
    return x**2**2

def sub3(x):
    return x-3

if __name__ == "__main__":

    #multiprocessing.set_start_method('spawn')

    source = iter(range(2**6))

    M = multi_Manager(source, [mul2,sub3], [serial_a, serial_b], 
                      chunksize=10,
                      procs=8)
    M.run()

    print ("Finished gracefully!")
