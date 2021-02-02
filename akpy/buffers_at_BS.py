'''
Buffers as FIFO queues.
We do not care about the content but just count number of packets in buffers
and keep track of the age of each packet.
When buffer is full, the (incoming) packets are tail-dropped.
'''
#will not use a queue, for speed we don't care about the contents that are being buffered
#https://docs.python.org/3/library/queue.html#queue.SimpleQueue
#import queue

import numpy as np
import copy

class Buffers:

    #restrict to fixed size buffers - AK-TODO make it general: check if buffer_size is a vector or number and initialize properly
    def __init__(self, num_buffers=10, buffer_size=8, max_packet_age=-1):
        self.__version__ = "0.1.0"
        self.num_buffers = num_buffers #number of buffers
        self.max_packet_age = int(max_packet_age) #limits the reported packet age to this cap value, but
        #to limit the number of bits to reprensent it, but the internal array still keeps the correct values
        self.time_counter = 0 #count discrete-time
        self.age_value_when_buffer_is_empty = int(0) #we have already used -1 (to look nicer) and 0
        #now effectively create buffers
        self.create_buffers_same_size(buffer_size)

    def create_buffers(self, buffer_sizes):
        if np.prod(buffer_sizes.shape) != self.num_buffers:
            raise Exception('Wrong size')
        self.buffer_sizes =  copy.deepcopy(buffer_sizes).flatten()
        self.buffer_occupancies = np.zeros((self.num_buffers,),np.uint64)
        #numpy is optimized for homogeneous arrays of numbers with fixed dimensions,
        #https://stackoverflow.com/questions/3386259/how-to-make-a-multidimension-numpy-array-with-a-varying-row-size
        #but I want to give support to buffers with different sizes, so I am going to
        #use a list of 1D numpy arrays instead of a 2D numpy array
        self.packets_arrival_time = list()
        for i in range(self.num_buffers):
            #use age_value_when_buffer_is_empty to represent initial state
            self.packets_arrival_time.append(self.age_value_when_buffer_is_empty*np.ones((self.buffer_sizes[i],),np.int))
        self.dropped_counter =  np.zeros((self.num_buffers,),np.uint64) #cummulative drops
        self.dropped_last_iteration = np.zeros((self.num_buffers,),np.uint64) #instantaneous drops

        # for buffer history
        self.buffer_history_incoming = [[0.] for i in range(self.num_buffers)]
        self.buffer_history_dropped = [[0.] for i in range(self.num_buffers)]

    def create_buffers_same_size(self, buffer_size):
        self.create_buffers(buffer_size * np.ones((self.num_buffers,),np.uint64))

    def reset_buffers(self):
        self.create_buffers(self.buffer_sizes)

    def packets_departure(self, num_candidate_packets_to_be_discarded):
        '''
        The oldest packet is in the beginning (left) of the array.
        So we need to shift to the left.
        Index i, buffer_occupancies[i],packets_arrival_time[i],dropped_counter[i]
        0 => 2 [1 2 0 0 0] 0
        1 => 4 [1 1 2 2 0] 0
        2 => 5 [1 1 1 2 2] 1
        Index i, buffer_occupancies[i],packets_arrival_time[i],dropped_counter[i]
        0 => 3 [1 2 3 0 0] 0
        1 => 5 [1 1 2 2 3] 1
        2 => 5 [1 1 1 2 2] 4
        '''
        num_candidate_packets_to_be_discarded = num_candidate_packets_to_be_discarded.flatten()
        num_removed_packets = copy.deepcopy(num_candidate_packets_to_be_discarded)
        for i in range(self.num_buffers):
            #removed_packets = num_discarded_packets[i]
            if num_removed_packets[i] > self.buffer_occupancies[i]:
                #raise Exception('Cannot extract ' + str(removed_packets) + ' which is more than buffer occupancy ' + str(self.buffer_occupancies[i]))
                #print('WARNING: Cannot extract ' + str(num_removed_packets[i]) + ' which is more than buffer occupancy ' + str(self.buffer_occupancies[i]))
                num_removed_packets[i] = self.buffer_occupancies[i]
                self.buffer_occupancies[i] = 0
            else:
                self.buffer_occupancies[i] -= num_removed_packets[i]
                #update packets_arrival_time
            this_packets_arrival_time = self.packets_arrival_time[i] #get numpy array from list
            num_remaining_packets = int(self.buffer_sizes[i] - num_removed_packets[i])
            this_packets_arrival_time[0:num_remaining_packets]=this_packets_arrival_time[num_removed_packets[i]:]
            this_packets_arrival_time[num_remaining_packets:]=self.age_value_when_buffer_is_empty #np.NaN #need to initialize the time for the earliest packets
            self.packets_arrival_time[i] = this_packets_arrival_time
        return num_removed_packets

    def get_dropped_last_iteration(self):
        return self.dropped_last_iteration

    def get_dropped_last_iteration_percent(self):
        return self.dropped_last_iteration_percentual

    def packets_arrival(self, num_incoming_packets):
        num_incoming_packets = num_incoming_packets.flatten()
        self.time_counter += 1 #update the counter
        self.dropped_last_iteration = np.zeros((self.num_buffers,), np.uint64)
        self.dropped_last_iteration_percentual = np.zeros((self.num_buffers,), np.float)
        for i in range(self.num_buffers):
            space_available = self.buffer_sizes[i] - self.buffer_occupancies[i]
            #check if there is space
            if space_available < 0:
                #all new packets are dropped
                self.dropped_last_iteration[i] = num_incoming_packets[i]
                self.dropped_counter[i] += num_incoming_packets[i]
                continue #there's nothing else to be done
            #we have space for at least some packets
            added_packets = np.minimum(num_incoming_packets[i], space_available)
            #first update the packets_arrival_time
            this_packets_arrival_time = self.packets_arrival_time[i] #get numpy array from list
            first_index = int(self.buffer_occupancies[i])
            last_index = int(first_index + added_packets)
            #use -1 to indicate it's the current time (given that we already updated it)
            this_packets_arrival_time[first_index:last_index]=self.time_counter-1
            self.buffer_occupancies[i] += added_packets
            if added_packets != num_incoming_packets[i]:
                #there is no space for all packets, discard some upcoming packets
                self.dropped_last_iteration[i] = num_incoming_packets[i] - added_packets
                self.dropped_counter[i] += self.dropped_last_iteration[i]
                #computing the percentual of discarded packets
                self.dropped_last_iteration_percentual[i] = self.dropped_last_iteration[i] / num_incoming_packets[i]

        self.save_history(num_incoming_packets, self.dropped_last_iteration)

    def oldest_packet_per_buffer(self):
        '''
        For each buffer, return for the oldest packet, its number of time instants buffered
        with a provided maximum (cap) value. If buffer is empty, indicate with age_value_when_buffer_is_empty.
        The oldest packet is in the beginning (left) of the array.
        Index i, buffer_occupancies[i],packets_arrival_time[i],dropped_counter[i]
        0 => 2 [1 2 0 0 0] 0
        1 => 4 [1 1 2 2 0] 0
        2 => 5 [1 1 1 2 2] 1
        Index i, buffer_occupancies[i],packets_arrival_time[i],dropped_counter[i]
        0 => 3 [1 2 3 0 0] 0
        1 => 5 [1 1 2 2 3] 1
        2 => 5 [1 1 1 2 2] 4
        '''
        interval_in_buffer = np.zeros((self.num_buffers,),np.int)
        for i in range(self.num_buffers):
            if self.buffer_occupancies[i] == 0: #buffer is empty, indicate with age_value_when_buffer_is_empty
                interval_in_buffer[i] = self.age_value_when_buffer_is_empty
                continue
            this_packets_arrival_time = self.packets_arrival_time[i] #get numpy array from list
            this_packet_age = int(self.time_counter - this_packets_arrival_time[0])
            if self.max_packet_age==-1:
                #no need to use maximum value
                interval_in_buffer[i] = this_packet_age
            else:
                interval_in_buffer[i] = np.minimum(this_packet_age, self.max_packet_age)

        return interval_in_buffer

    def pretty_print(self):
        oldest_packet_per_buffer = self.oldest_packet_per_buffer()
        print('Index i, buffer_occupancies[i],packets_arrival_time[i],dropped_counter[i],oldest_packet_per_buffer[i]')
        for i in range(self.num_buffers):
            print(i,'=>',self.buffer_occupancies[i],self.packets_arrival_time[i],
                  self.dropped_counter[i], oldest_packet_per_buffer[i])

    def get_buffer_states(self):
        return (self.buffer_occupancies, self.oldest_packet_per_buffer())

    def save_history(self, num_incoming_packets, dropped):
        for i in range(self.num_buffers):
            self.buffer_history_incoming[i] += num_incoming_packets[i]
            self.buffer_history_dropped[i] += dropped[i]

if __name__ == '__main__':
    #test buffers with 3 arrivals and 1 departure
    buffers = Buffers(num_buffers=3, max_packet_age=-1)
    buffer_size = 5
    buffers.create_buffers_same_size(buffer_size)
    num_input_packets = np.array([1,2,3])
    buffers.packets_arrival(num_input_packets)
    buffers.pretty_print()
    buffers.packets_arrival(num_input_packets)
    buffers.pretty_print()
    buffers.packets_arrival(num_input_packets)
    buffers.pretty_print()
    num_discarded_packets = 2*np.array([1,1,1])
    buffers.packets_departure(num_discarded_packets)
    buffers.pretty_print()
    num_discarded_packets = np.array([1,2,2])
    buffers.packets_departure(num_discarded_packets)
    buffers.pretty_print()
