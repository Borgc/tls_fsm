import graphviz
import random
import time
import matplotlib.pyplot as plt


class FiniteStateMachine:
    def __init__(self, description_file):
        self.states = set()
        self.inputs = set()
        self.outputs = set()
        self.initial_state = None
        self.transitions = []
        self.current_state = None
        
        self.parse_description(description_file)
        self.create_graph()

    def parse_description(self, filename):
        with open(filename, 'r') as file:
            lines = file.readlines()

        # Parse header
        assert lines[0].startswith('F'), "Invalid file format"
        self.num_states = int(lines[1].split()[1])
        self.num_inputs = int(lines[2].split()[1])
        self.num_outputs = int(lines[3].split()[1])
        self.initial_state = int(lines[4].split()[1])

        # Parse transitions
        self.num_transitions = int(lines[5].split()[1])
        for i in range(6, 6 + self.num_transitions):
            line = lines[i].split()
            start_state = int(line[0])
            input_symbol = int(line[1])
            self.inputs.add(input_symbol)
            end_state = int(line[2])
            output_symbol = int(line[3])
            self.outputs.add(output_symbol)
            self.transitions.append((start_state, input_symbol, end_state, output_symbol))

    def step(self, input_symbol):
        output_symbol = None

        for transition in self.transitions:
            start_state, transition_input, end_state, transition_output = transition
            if self.current_state == start_state and input_symbol == transition_input:
                self.current_state = end_state
                output_symbol = transition_output
                break

        return self.current_state, output_symbol
    
    def gen_random_seq(self, n):
        """Generates random input string length of n

        Args:
            n (_int_): _desireble sequence length_
        """
        seq = []
        inputs = list(self.inputs)
        for i in range(n):
            seq.append(random.choice(inputs))
        return seq
        
    def input_sequence(self, seq):
        """Feeds the sequence to the machine with step() function

        Args:
            seq (_list_): _description_
        """
        self.current_state = self.initial_state
        
        output = []
        for i in range(len(seq)):
            output.append(self.step(seq[i]))
        return output
    
    def mutate(self, mutation_type):
        """
        mutation types:
        t: Error in transition
        o: Error in output
        s: Error in extra state
        """
        if mutation_type == 't':  
            self._mutate_transition_error()
        elif mutation_type == 'o':  
            self._mutate_output_error()
        elif mutation_type == 's':  
            self._mutate_extra_state_error()
        else:
            raise ValueError("Invalid mutation type")
        
    def _mutate_transition_error(self):
        transition_index = random.randint(0, len(self.transitions) - 1)
        start_state, input_symbol, end_state, output_symbol = self.transitions[transition_index]
        new_end_state = random.randint(0, self.num_states - 1)
        while new_end_state == end_state:
            new_end_state = random.randint(0, self.num_states - 1)
        self.transitions[transition_index] = (start_state, input_symbol, new_end_state, output_symbol)

    def _mutate_output_error(self):
        transition_index = random.randint(0, len(self.transitions) - 1)
        start_state, input_symbol, end_state, output_symbol = self.transitions[transition_index]
        new_output_symbol = random.randint(0, self.num_outputs - 1)
        while new_output_symbol == output_symbol:
            new_output_symbol = random.randint(0, self.num_outputs - 1)
        self.transitions[transition_index] = (start_state, input_symbol, end_state, new_output_symbol)

    def _mutate_extra_state_error(self):
        state_index = random.randint(0, self.num_states - 1)
        new_state_index = self.num_states
        self.num_states += 1
        self.states.add(new_state_index)

        for transition in self.transitions:
            start_state, input_symbol, end_state, output_symbol = transition
            if start_state == state_index:
                self.transitions.append((new_state_index, input_symbol, end_state, output_symbol))
                self.num_transitions += 1
                
        for transition in self.transitions:
            start_state, input_symbol, end_state, output_symbol = transition
            if end_state == state_index:
                self.transitions.remove(transition)
                self.transitions.append((start_state, input_symbol, new_state_index, output_symbol))
                break
        
        for transition in self.transitions:
            start_state, input_symbol, end_state, output_symbol = transition
            if start_state == new_state_index:
                transition = start_state, input_symbol, random.randint(0, self.num_states - 1), output_symbol
                break
        

    def create_graph(self):
        dot = graphviz.Digraph(comment='Finite State Machine')

        # Add states
        for state in range(self.num_states):
            if state == self.initial_state:
                dot.node(str(state), f"State {state}", shape='doublecircle')
            else:
                dot.node(str(state), f"State {state}", shape='circle')

        # Add transitions
        for transition in self.transitions:
            start_state, input_symbol, end_state, output_symbol = transition
            dot.edge(str(start_state), str(end_state), f"In: {input_symbol}, Out: {output_symbol}")

        self.graph = dot

# Usage example
def test(n, fsm, fsm_mutated):
    """simple test

    Args:
        n (_int_): _input length_,
        error_type (_str_): 
        _t - transition error, 
        o - output error, 
        s - state error_
        fsm - etalon fsm
    """

    seq = fsm.gen_random_seq(n)
    out1 = fsm.input_sequence(seq)
    out2 = fsm_mutated.input_sequence(seq)
    if out1 != out2:
        return 1
    else:
        return 0

    

    #fsm_mutated.create_graph()
    #fsm_mutated.graph.render('fsm_mutated_diagram', format='png')
    

def multiple_test(n, num, fsm, fsm_mutated):
    """test with multiple input sequence for same mutant

    Args:
        n (_int_): _input length_,
        error_type (_str_): 
        _t - transition error, 
        o - output error, 
        s - state error_
        fsm - etalon fsm
    """

    errors = 0
    for i in range(num):
        seq = fsm.gen_random_seq(n)
        out1 = fsm.input_sequence(seq)
        out2 = fsm_mutated.input_sequence(seq)
        
        
        
        if out1 != out2:
            # print(f"out1 = {out1}")
            # print(f"out2 = {out2}")
            errors +=1
    return errors /num
        
            
    
    
def stat_test(n, error_type):
    """count statistics about multiple simple tests

    Args:
        k (_int_): _amount of tests_
        error_type (_str_): _'t', 'o', 's'_
    """
    #num of FSMs
    res = 0
    for j in range(100):
        
        fsm = FiniteStateMachine(f"G:\\coursework\\FSMs\\{j}.fsm")
        for i in range(100): #num of errors for each automata
            fsm_mutated = FiniteStateMachine(f'G:\\coursework\\FSMs\\{j}.fsm')
            fsm_mutated.mutate(error_type)
            #fsm.create_graph()
            #fsm.graph.render('fsm_diagram', format='png')
            
            res += test(n, fsm, fsm_mutated)
    print(f"stat test: {res / (100 * 100)}") 

def multiple_stat_test(n, num, error_type):
    """count statistics about multiple tests with multiple input sequences for each mutant

    Args:
        k (_int_): _description_
        error_type (_str_): _'t', 'o', 's'_

    """
    res = 0
    for j in range(100): #num of automatas
        fsm = FiniteStateMachine(f"G:\\coursework\\FSMs\\{j}.fsm")
        for i in range(10): #num of errors for each automata
            fsm_mutated = FiniteStateMachine(f'G:\\coursework\\FSMs\\{j}.fsm')
            fsm_mutated.mutate(error_type)
            #fsm.create_graph()
            #fsm.graph.render('fsm_diagram', format='png')
            res += multiple_test(n, num, fsm, fsm_mutated)
    return res / (1000)

def experiment(n, num, error_type):
    
    print(f"input sequence len: {n}")
    print(f"number of input sequences: {num}")
    print(f"error_type: {error_type}")
    start_time = time.time()
    res = multiple_stat_test(n, num, error_type)
    print(f"errors detected : {res}")
    end_time = time.time()
    print(f"test time: {end_time - start_time}")
    return res
        
if __name__ == "__main__":
    # inputs = [[1, 500],
    #            [2, 250],
    #            [4, 125],
    #            [5, 100],
    #            [10, 50],
    #            [20, 25],
    #            [25, 20],
    #            [50, 10],
    #            [100, 5],
    #            [125, 4],
    #            [250, 2],
    #            [500, 1]
    #           ]
    # for i in range(len(inputs)):
    #     n = inputs[i][0]
    #     num = inputs[i][1]
        
    #     experiment(n, num, 't')
    # for i in range(len(inputs)):
    #     n = inputs[i][0]
    #     num = inputs[i][1]
    
    #     experiment(n, num, 'o')
    # for i in range(len(inputs)):
    #     n = inputs[i][0]
    #     num = inputs[i][1]
        
    #     experiment(n, num, 's')
    
    
#     inputs = [[1, 1],
#              [1, 1],
#              [2, 1],
#              [4, 1],
#              [5, 1],
#              [10, 1],
#              [20, 1],
#              [25, 1],
#              [50, 1],
#              [100, 1],
#              [125, 1],
#              [250, 1],
#              [500, 1],
#              [625, 1],
#              [750, 1],
#              [875, 1],
#              [1000, 1]
            
#             ]
#     input_sequence_lengths = []
#     t_errors_detected = []
#     o_errors_detected = []
#     s_errors_detected = []
#     for i in range(len(inputs)):
#         input_sequence_lengths.append(inputs[i])
        
#     for i in range(len(inputs)):
#         n = inputs[i][0]
#         num = inputs[i][1]
        
#         t_errors_detected.append(experiment(n, num, 't'))
#     for i in range(len(inputs)):
#         n = inputs[i][0]
#         num = inputs[i][1]
    
#         o_errors_detected.append(experiment(n, num, 'o'))
#     for i in range(len(inputs)):
#         n = inputs[i][0]
#         num = inputs[i][1]
#         s_errors_detected.append(experiment(n, num, 's'))
#     print(len(inputs), len(t_errors_detected), len(o_errors_detected), len(s_errors_detected))
#     print(input_sequence_lengths, t_errors_detected, o_errors_detected, s_errors_detected)

    

# # Create the figure and axis
# fig, ax = plt.subplots()

# # Plot the data
# ax.plot(input_sequence_lengths, t_errors_detected, label='Type t')
# ax.plot(input_sequence_lengths, o_errors_detected, label='Type o')
# ax.plot(input_sequence_lengths, s_errors_detected, label='Type s')

# # Set the title and labels
# ax.set_title('Errors Detected vs. Input Sequence Length')
# ax.set_xlabel('Input Sequence Length')
# ax.set_ylabel('Errors Detected')

# # Add a legend
# ax.legend()
# plt.savefig("detected_errors_20.png")
# # Show the plot
# plt.show()
#     
    tls = FiniteStateMachine(f'G:\\diplom\\simple_tls.fsm')
    tls.create_graph()
    tls.graph.render('fsm_diagram', format='png')
        