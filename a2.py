import numpy as np
# ^ is the xor operator 
# Define the output function as per the diagram
def output(s0, s1, i):
    P1 = i ^ s1
    P2 = i ^ s0
    P3 = s1 ^ s0
    return [P1,P2,P3]

def encoder(bitstream):
    # Step 1 add 2 leading zeros to simulate state
    bitstream = np.concatenate(([False, False], bitstream))
    # Initialize and empty endcoded away 
    encoded = []
    # Iterate over the original length of th input.
    for n in range(len(bitstream)-2):
        # Concatenate the output bits
        encoded.extend(output(bitstream[n], bitstream[n+1], bitstream[n+2]))
    # remove the added state 0 bits
    bistream = bitstream[2:]
    return encoded

def time_instance(n_states):
    # Wrapper of List of Dictionaries 
    # cost is initialized as infinity 
    # prev holds the index of the previous 
    # node and the decoded bit.
    return [{'cost': np.inf, 'prev': [None,None]} for _ in range(n_states)]

def machine(input, current):
    nexts = ['00', '00', '01', '01'] if input == 0 else ['10', '10', '11', '11']
    return nexts[current]

def Hamming_dist(bit1, bit2): return bit1 ^ bit2

states = {'00': 0, '01':1, '10':2, '11':3}
#states_inv = {0: [False,False], 1: [False,True], 2: [True,False], 3: [True,True]}
states_inv = {0: [0,0], 1: [0,1], 2: [1,0], 3: [1,1]}
def forward_decoder(encoded):
    # Initialize the empty trellis diagram
    # Has length as the bitstream + 1 for the initial state
    instances = [time_instance(4) for _ in range(int(len(encoded)/3)+1)]
    # Initialize the initial state 
    instances[0][0]['prev'] = [0, 0]
    instances[0][0]['cost'] = 0
    # loop over the timesteps
    for i in range(len(instances)-1):
        # The observed (noisy output)
        observed_op = encoded[i*3 : (i+1)*3]
        # loop over the states
        for state in range(4):
            # If the state hasn't been reached break
            if instances[i][state]['prev'][0] is None:
                continue
            # calcutate the next state index
            next0state, next1state = states[machine(0,state)], states[machine(1,state)]
            # get the bits of the current state
            state_bits = states_inv[state]
            # use the state bits with 0 and 1 and get the output of the state-bit pair
            op = [output(state_bits[1], state_bits[0], j) for j in [0,1]]
			# Calculate the cost(Hamming dist) between the output and the observed output
            cost = [sum(list(map(Hamming_dist,observed_op,op[j]))) for j in [0,1]]
			# Update next bits if 0 bit and if 1 bit
            # Update only if the current cost is less that the cost already
            # in the next state node
            # for 0 bit (dashed line)
            if instances[i+1][next0state]['cost'] > cost[0] + instances[i][state]['cost']:
                instances[i+1][next0state]['prev'] = [state, 0]
                instances[i+1][next0state]['cost'] = cost[0] + instances[i][state]['cost']
			# repeat for the 1 bit  (solid line)
            if instances[i+1][next1state]['cost'] > cost[1] + instances[i][state]['cost']:
                instances[i+1][next1state]['prev'] = [state, 1]
                instances[i+1][next1state]['cost'] = cost[1] + instances[i][state]['cost']
    return instances
def viterbi_decode(instances):
    # Get the minimum state based on cost at the last instance
    min_state = min(instances[len(instances) - 1], key = lambda x: x['cost'])
    # Get previous state and decoded bit
    prev_state = instances[len(instances) - 2][min_state['prev'][0]]
    bit = min_state['prev'][1]
    # Initialize the result array
    result = [bit]
    # Loop over the middle time instances
    for inst in range(len(instances)-2,0,-1):
        # Get previous state and decoded bit
        bit = prev_state['prev'][1]
        prev_state = instances[inst -1][prev_state['prev'][0]]
        # Append to the result
        result.append(bit)
    # Return the revered array.
    return result[::-1]
def decode(encoded): return viterbi_decode(forward_decoder(encoded))


def binarize(image):
    output = []
    for row in image:
        for pixel in row:
            output.extend(list(map(bool,list(map(lambda x: int(x), list(f"{pixel:08b}"))))))
    return np.array(output, dtype= bool)
            

def de_binarize(bitstream, rows, cols):
    image = np.zeros((rows,cols), dtype = np.uint8)
    for pixel in range(rows*cols):
        a = np.array(bitstream[pixel*8: pixel*8 + 8], dtype= np.int)
        intermediate = int("".join(str(x) for x in a), 2)
        image[np.int(pixel/cols), pixel%cols] = intermediate
    return image

def rmse(source, decoded):
    return np.sqrt(np.mean(np.square(source - decoded)))