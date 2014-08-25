import multiprocessing, time, traceback, logging

logger = logging.getLogger()
logging.basicConfig()
logger.setLevel(logging.DEBUG)

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
            #logger.debug("{} is waiting".format(proc_name))
            next_task = self.task_queue.get()
            logger.debug("{} has got {}".format(proc_name, next_task))
            #logger.debug("{} Poison status: {}".format(proc_name,
            #                                    self.ate_poison(next_task)))

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

        if procs == None:
            self.procs = multiprocessing.cpu_count()
        else:
            self.procs = procs

        self.source = source
        self.func_T = TASK_CHAIN
        self.func_S = SERIAL_CHAIN

        assert(len(TASK_CHAIN) == len(SERIAL_CHAIN))

        k = len(TASK_CHAIN)

        self.T_input  = [multiprocessing.Queue() for _ in range(k)]
        self.S_input  = [multiprocessing.Queue() for _ in range(k)]
        self.S_output = [multiprocessing.Queue() for _ in range(k)]

        self.all_Q = [self.T_input,self.S_input,self.S_output]

        self.C = [[Consumer(t,q) for _ in range(self.procs)] 
                  for t,q in zip(self.T_input,self.S_input)]

        self.queue_T_input_closed  = [0 for _ in xrange(k)]
        self.queue_S_input_closed  = [0 for _ in xrange(k)]
        self.queue_S_output_closed = [0 for _ in xrange(k)]

        # Start consumers
        for c in self.consumer_iter():
            c.start()

        #class is_consumer_level_alive:
        #    def __init__(self, level):
        #        self.level = level
        #    def __call__(self):
        #        for c in C[level]:
        #            if not c.is_alive():
        #                return False
        #    return True

        self._empty_source = False
        self._is_complete  = False

        # This is the order things get shutdown
        # the inital input is always mapped to the source
        #
        #self.close_mapping = {self.T_input[0] : (lambda x: self._empty_source)}
        #for k,TX in enumerate(self.T_input):
        #    if k>0:
        #        self.close_mapping[TX] = is_consumer_level_alive(k)
        #print self.close_mapping
        #exit()
        

    def consumer_iter(self):
        for C_block in self.C:
            for c in C_block: yield c

    def run(self):
        while not self._is_complete:
            self.cycle()
            time.sleep(self.cycle_sleep)


    def is_complete(self): 
        return self._is_complete
    
    def process_serial(self):
        for consumers, S_in,S_out,f in zip(self.C,
                                           self.S_input, 
                                           self.S_output, 
                                           self.func_S):
            while S_in.qsize():
                val    = S_in.get()
                result = f(val)
                S_out.put(result)

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

    def _close_queue(self, Q):
        for _ in range(self.procs):
            Q.put(PoisonPill())
        Q.close()
        Q.join_thread()       

    def shutdown(self):
        print "SHUTDOWN *****************"
        for k in xrange(len(self.T_input)):
            print "IS ALIVE?", self.C[k][0].is_alive()
            if self.C[k][0].is_alive():

                logger.debug( "Empty status: %s" % [[Q.qsize() for Q in QX] for QX in self.all_Q] )
                self._close_queue( self.T_input[k]) 
                self._close_queue( self.S_input[k] )
                self._close_queue( self.S_output[k] )
                logger.debug( "Empty status: %s" % [[Q.qsize() for Q in QX] for QX in self.all_Q] )
                for c in self.C[k]:
                    c.join()

                print "BREAKING!"
                return False

                    
            logger.debug( "Empty status: %s" % [[Q.qsize() for Q in QX] for QX in self.all_Q] )
        print "SHUTDOWN COMPLETE *************"
        self._is_complete = True
        return True
    
    def is_empty(self):
        ''' Check if all queues are empty, including source '''
        if not self._empty_source: return False

        logger.debug( "Empty status: %s" % [[Q.qsize() for Q in QX] for QX in self.all_Q] )
        for QX in [self.T_input,self.S_input,self.S_output]:
            for Q in QX:
                if Q.qsize() > 0: return False

        return True

    def cycle(self):
        self.process_serial()
        free = self.count_free_slots()
        #print free, "slots open"
        if not free: return False

        # Pulling anything off of the last serial queue and discard it
        while not self.S_output[-1].empty():
            x = self.S_output[-1].get()

        # Step through queues backwards
        # ignore first T and last S
        for consumers, S_out,T_in,f in zip(
                self.C[:1],
                self.S_output[:1],
                self.T_input[1:],
                self.func_T[1:]):
            
            if consumers[0].is_alive():
                while not S_out.empty():
                    val  = S_out.get()
                    task = Generic_Task(f, val)
                    T_in.put(task)
                    free -= 1
                    if not free: return False

        # Pull from the source if still here
        for n in range(free):
            val = self.pull_source()
            task = Generic_Task(self.func_T[0], val)
            if self._empty_source:
                logger.debug("Source is now empty")
                break

            self.T_input[0].put( task )

        if self.is_empty():
            self.shutdown()


# These test methods have to be global to work with 'spawn' method!

def serial_a(x):
    #print ("A result: ", x)
    return x

def serial_b(x):
    #print ("B result: ", x)
    return x

def mul2(x):
    return x**2**2

def sub3(x):
    return x-3

if __name__ == "__main__":

    #multiprocessing.set_start_method('spawn')

    source = iter(range(10**3))

    M = multi_Manager(source, [mul2,sub3], [serial_a, serial_b], 
                      chunksize=10,
                      procs=4)
    M.run()

    print ("Finished gracefully!")
