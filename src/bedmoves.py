import numpy as np
import ciw
import random
import tqdm
import pandas as pd

class NullLock(object):
    def __init__(self, dummy_resource=None):
        self.dummy_resource = dummy_resource
    def __enter__(self):
        return self.dummy_resource
    def __exit__(self, *args):
        pass

max_capacities = (3, 2, 2, 3, 2, 2, 1, 1, 1)

empty_state = np.array(
    (
        (0, 0, 0, 0, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0, 0, 0, 0)
    ), dtype=int
)

def get_resource_use_per_time_unit(state):
    """
    Calculates the resource use for a given state per time unit

    - One FTE per block containing at least one green patient
    - One FTE per amber patient
    - One FTE per red patient

    Arguments:
      - `state` a 9x3 matrix of integers {0, 1, 2, 3} representing
         the state of the ward.

    Returns: and integer number of resources used per time unit.
    """
    green = (state[0,:] > 0).sum()
    amber = state[1,:].sum()
    red = state[2,:].sum()
    return green + amber + red


def get_penalty_per_time_unit(state, isolation_penalty):
    """
    Calculates the penalty for having isolation patients in a
    general block

    Arguments:
      - `state` a 9x3 matrix of integers {0, 1, 2, 3} representing
         the state of the ward.
      - `isolation_penalty`: the numerical penalty patient per
         time unit of not being in an isolation ward.

    Returns: a numerical penalty per time unit for the given state.
    """
    return state[2,:-3].sum() * isolation_penalty



def get_move_penalty(from_block, to_block, b1, b2):
    """
    Calculates the instantaneous penalty associated with a bed move.

    Arguments:
      - `from_block`: the block the patient was moved from
      - `to_block`: the block the patient is moved to
      - `b1`: the penalty for moving to an adjacent block
              (representing not a move, but a penalty for
              stretching resources across blocks)
      - `b2`: the penalty for moving to a non-adjacent block

    Returns: a numerical penalty for a bed move.
    """
    A = [
        [0, b1, b2, b1, b2, b2, b2, b2, b2],
        [b1, 0, b1, b2, b1, b2, b2, b2, b2],
        [b2, b1, 0, b2, b2, b1, b2, b2, b2],
        [b1, b2, b2, 0, b1, b2, b2, b2, b2],
        [b2, b1, b2, b1, 0, b1, b2, b2, b2],
        [b2, b2, b1, b2, b1, 0, b2, b2, b2],
        [b2, b2, b2, b2, b2, b2, 0, b2, b2],
        [b2, b2, b2, b2, b2, b2, b2, 0, b2],
        [b2, b2, b2, b2, b2, b2, b2, b2, 0]
    ]
    return A[from_block - 1][to_block - 1]


def move_patient(state, patient_type, from_block, to_block):
    """
    Returns the state that results from moving a patient.

    Arguments:
      - `state` a 9x3 matrix of integers {0, 1, 2, 3} representing
         the state of the ward.
      - `patient_type`: the type of the patient being moved, either
         2: 'red', 1: 'amber', or 0: 'green'
      - `from_block`: the block the patient was moved from
      - `to_block`: the block the patient is moved to
    """
    new_state = state.copy()
    new_state[patient_type, from_block - 1] -= 1
    new_state[patient_type, to_block - 1] += 1
    return new_state


def insert_patient(state, patient_type, to_block):
    """
    Returns the state that results from inserting a patient.

    Arguments:
      - `state` a 9x3 matrix of integers {0, 1, 2, 3} representing
         the state of the ward.
      - `patient_type`: the type of the patient being moved, either
         2: 'red', 1: 'amber', or 0: 'green'
      - `to_block`: the block the patient is moved to
    """
    new_state = state.copy()
    new_state[patient_type, to_block - 1] += 1
    return new_state


def remove_patient(state, patient_type, from_block):
    """
    Returns the state that results from removing a patient.

    Arguments:
      - `state` a 9x3 matrix of integers {0, 1, 2, 3} representing
         the state of the ward.
      - `patient_type`: the type of the patient being moved, either
         2: 'red', 1: 'amber', or 0: 'green'
      - `from_block`: the block the patient was moved from
    """
    new_state = state.copy()
    new_state[patient_type, from_block - 1] -= 1
    return new_state


def get_available_insert_moves(state):
    """
    Lists all available places where a patient can be inserted.

    Arguments:
      - `state` a 9x3 matrix of integers {0, 1, 2, 3} representing
         the state of the ward.

    Returns: a list of blocks that the patient can be inserted.    
    """
    available_moves = []
    for i, capacity in enumerate(max_capacities):
        free_beds = capacity - state[:,i].sum()
        if free_beds > 0:
            available_moves.append(i + 1)
    return available_moves


def get_available_moves(state):
    """
    Lists all available bed moves possible.

    Arguments:
      - `state` a 9x3 matrix of integers {0, 1, 2, 3} representing
         the state of the ward.

    Returns: a list of possible moves, that is tuples (a, b, c),
        where a is the patient type, b is where they move from,
        and c is where they can move to.
    """
    available_moves = []
    available_inserts = get_available_insert_moves(state)
    for patient_type in range(3):
        from_blocks = np.where(state[patient_type,:] > 0)[0]
        for from_block in from_blocks:
            for to_block in available_inserts:
                if from_block + 1 != to_block:
                    available_moves.append((patient_type, from_block + 1, to_block))
    return available_moves


def combine_Qvalues(list_of_Qval_dfs):
    """
    Combines many dataframes of Qvales (the output of Q.output_Q_values),
    such that the resulting Qvalues are the weighted average of the
    Qvalues across all dataframes (weighted by number of hits), and the
    number of hits are summed.

    Arguments:
      + `list_of_Qval_dfs`: a list of dataframes to be combined

    Returns: A combined dataframe.
    """
    concated_df = pd.concat(list_of_Qval_dfs)
    concated_df['Q x hits'] = concated_df['Q'] * concated_df['hits'].clip(lower=1)
    combined_df = pd.DataFrame(
        {
            'Q': concated_df.groupby(concated_df.index)['Q x hits'].sum(),
            'hits': concated_df.groupby(concated_df.index)['hits'].sum(),
        }
    )
    combined_df['Q'] = combined_df['Q'] / combined_df['hits'].clip(lower=1)
    del combined_df['hits']

    combined_df.index.name = None
    return combined_df


class EpsilonHard:
    def __init__(self, epsilon, QLearning):
        """
        Initialises the epsilon-hard action selection policy object.
        That is, chooses the best strategy `epislon` on the time, and
        randomly otherwise.

        Arguments:
          + `epsilon`: a probability (low: explore more, high: exploit more)
          + `QLearning`: a Qlearning object.
        """
        self.epsilon = epsilon
        self.QLearning = QLearning

    def choose_arriving_block(self, state, patient_type):
        """
        Randomly chooses a block for an arriving patient 1-epsilon
        of the time. Otherwise chooses the best.
        """
        available_blocks = get_available_insert_moves(state)
        if len(available_blocks) == 0:
            return False
        if random.random() < self.epsilon:
            return self.choose_best_block(state, patient_type, available_blocks)
        return self.choose_random_block(available_blocks)

    def choose_random_block(self, available_blocks):
        """
        Chooses a block randomly from a list of blocks.

        Arguments:
          + `available_blocks`: a list of available blocks to
             insert a patient to.

        Returns: a block.
        """
        return random.choice(available_blocks)

    def choose_best_block(self, state, patient_type, available_blocks):
        """
        Chooses the best action.
        """
        next_qstates_as = [(a, self.QLearning.get_hash_state((state, patient_type), a)) for a in available_blocks]
        next_Qs = [(qstatea[0], self.QLearning.Qvals_dict.get(qstatea[1], 0.0)) for qstatea in next_qstates_as]
        random.shuffle(next_Qs)
        return max(next_Qs, key=lambda x: x[1])[0]

    def __repr__(self):
        return f"EpsilonHard-{self.epsilon}"



class QLearning:
    def __init__(self, learning_rate, discount_factor, transform_parameter, initial_Qvalues=None, learn=True):
        """
        Initialises the QLearning object.

        Arguments:
          - `learning_rate`: the learning rate of the Q-learning algorithm (a number between 0 and 1)
          - `discount_factor`: the discount factor of the Q-learning algorithm (a number between 0 and 1)
          - `transform_parameter`: a parameter to transform costs into rewards via e^{-transform_parameter * cost}
          - `initial_Qvalues`: a dataframe of Q-values
        """
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.transform_parameter = transform_parameter
        self.previous_cost = 0.0
        self.hash_state = None
        self.learn = learn
        self.initialise_df(initial_Qvalues)
        self.Qvals_dict = self.Qvals_df['Q'].to_dict()
        self.Qhits_dict = self.Qvals_df['hits'].to_dict()

    def __repr__(self):
        return "QLearning"

    def initialise_df(self, initial_Qvalues):
        """
        Initialises Qvals dataframe as either empty or with some
        previously learned Q-values.
        """
        if initial_Qvalues is None:
            self.Qvals_df = pd.DataFrame({'Q': [], 'hits': []})
        else:
            self.Qvals_df = initial_Qvalues.copy()
            self.Qvals_df['hits'] = 0

    def attach_simulation(self, simulation):
        """
        Attaches the simulation object to the QLearning object
        """
        self.simulation = simulation

    def get_hash_state(self, state, action):
        """
        Returns a hashable version of the state.
        """
        return "((" + ",".join(["(" + ",".join(map(str, state_row)) + ")" for state_row in state[0]]) + ")," + str(state[1]) + ")-" + str(action)

    def update_Q_values(self, next_state, next_action):
        """
        Updates the Q-values according to the Q-learning update:
        """
        if self.learn:
            cost = self.simulation.overall_cost - self.previous_cost
            self.previous_cost = self.simulation.overall_cost
            R = self.transform_cost(cost=cost)
    
            if self.hash_state is not None:
                stateaction = self.hash_state
                oldQ = self.Qvals_dict.get(stateaction, 0.0)
                newQ = (
                    ((1 - self.learning_rate) * oldQ)
                    + (self.learning_rate * (
                        R + (
                            self.discount_factor * self.get_best_future_reward(next_state=next_state)
                        )
                    ))
                )
                self.Qvals_dict[stateaction] = newQ
                self.Qhits_dict[stateaction] = self.Qhits_dict.get(stateaction, 0) + 1

            self.hash_state = self.get_hash_state(next_state, next_action)

    def transform_cost(self, cost):
        """
        Transforms the cost (C) into a reward (R) via:

        R = e^{-p * C}

        where p is the transform parameter.
        """
        return np.exp(-self.transform_parameter * cost)

    def get_best_future_reward(self, next_state):
        """
        Returns the maximum future reward if taking the optimal action
        when in the future state.
        """
        available_inserts = get_available_insert_moves(state=next_state[0])
        next_hash_states = [self.get_hash_state(next_state, a) for a in available_inserts]
        return max(self.Qvals_dict.get(hash_state, 0.0) for hash_state in next_hash_states)

    def update_Qvals_df(self):
        """
        Updates the Qvals_df with the newly learned Qvals_dict.
        """
        Qs = []
        hits = []
        indices = []
        for stateaction in self.Qvals_dict:
            indices.append(stateaction)
            Qs.append(self.Qvals_dict[stateaction])
            hits.append(self.Qhits_dict[stateaction])
        self.Qvals_df = pd.DataFrame(
            {'Q': Qs, 'hits': hits}, index=indices
        )


class Patient:
    def __init__(self, patient_type, los, arrival_date, block):
        """
        A class to keep track of a patient's length of stay.

        Arguments:
          - `patient_type`: the type of the patient being moved, either
             2: 'red', 1: 'amber', or 0: 'green'
          - `los`: a numeric length of stay
          - `now`: the current date
          - `block`: the block number they are currently in
        """
        self.patient_type = patient_type
        self.los = los
        self.arrival_date = arrival_date
        self.exit_date = self.arrival_date + self.los
        self.block = block


class BedMoveSimulation:
    def __init__(
        self,
        arrival_distributions,
        los_distributions,
        action_chooser,
        isolation_penalty,
        adjacent_move_penalty,
        nonadjacent_move_penalty,
        Qlearning,
        seed
    ):
        """
        Initialises the simulation object.

        Arguments:
          + `arrival_distributions`: a list of Ciw distribution objects
             representing the inter-arrival times of the green, amber,
             and red patients.
          + `los_distributions`: a list of Ciw distribution objects
             representing the length of stay times of the green, amber,
             and red patients.
          + `action_chooser`: an object that governs the choice fo actions.
          + `seed`: the random seed for the pseudorandom number generator.
          + `isolation_penalty`: the numerical penalty patient per time
             unit of not being in an isolation ward.
          + `adjacent_move_penalty`: the penalty for moving to an adjacent
             block (representing not a move, but a penalty for stretching
             resources across blocks)
          + `nonadjacent_move_penalty`: the penalty for moving to a
             non-adjacent block
        """
        self.arrival_distributions = arrival_distributions
        self.los_distributions = los_distributions
        self.action_chooser = action_chooser
        self.Qlearning = Qlearning
        self.Qlearning.attach_simulation(simulation=self)
        
        self.prev_now = 0.0
        self.now = 0.0
        self.overall_cost = 0.0

        self.isolation_penalty = isolation_penalty
        self.adjacent_move_penalty = adjacent_move_penalty
        self.nonadjacent_move_penalty = nonadjacent_move_penalty

        ciw.seed(seed)
        self.next_arrivals = {
            0: self.arrival_distributions[0].sample(),
            1: self.arrival_distributions[1].sample(),
            2: self.arrival_distributions[2].sample()
        }
        self.patients = []
        self.state = empty_state

    def find_next_arrival_date(self):
        """
        Returns the next date an arrival happens and
        what type of patient that arrival will be.
        """
        next_type = min(
            self.next_arrivals,
            key=self.next_arrivals.get
        )
        return self.next_arrivals[next_type], next_type

    def find_next_exit_date(self):
        """
        Returns the next date an exit happens and what
        patient is to exit.
        """
        if len(self.patients) == 0:
            return float('inf'), None
        next_patient = min(
            self.patients,
            key=lambda patient: patient.exit_date
        )
        return next_patient.exit_date, next_patient

    def simulate_until_max_time(self, max_time, lock=NullLock(), progress_bar=False, progress_bar_description=None):
        if progress_bar:
            with lock:
                self.progress_bar = tqdm.tqdm(total=max_time, desc=progress_bar_description)

        while self.now < max_time:
            next_arrival, patient_type = self.find_next_arrival_date()
            next_exit, patient = self.find_next_exit_date()
            if next_arrival < next_exit:
                self.arrival(next_arrival=next_arrival, patient_type=patient_type)
            else:
                self.exit(patient=patient)

            if progress_bar:
                with lock:
                    remaining_time = max_time - self.progress_bar.n
                    time_increment = self.now - self.prev_now
                    self.progress_bar.update(min(time_increment, remaining_time))

        if progress_bar:
            with lock:
                remaining_time = max(max_time - self.progress_bar.n, 0)
                self.progress_bar.update(remaining_time)
                self.progress_bar.close()

    def arrival(self, next_arrival, patient_type):
        """
        Generates a patient and decides where the patient should go.
        """
        self.next_arrivals[patient_type] += self.arrival_distributions[patient_type].sample()
        los = self.los_distributions[patient_type].sample()
        to_block = self.action_chooser.choose_arriving_block(state=self.state, patient_type=patient_type)
        if to_block is not False:
            self.inflict_cost(update_time=next_arrival)
            self.prev_now = self.now
            self.now = next_arrival
            self.Qlearning.update_Q_values(next_state=(self.state, patient_type), next_action=to_block)
            arriving_patient = Patient(patient_type=patient_type, los=los, arrival_date=self.now, block=to_block)
            self.patients.append(arriving_patient)
            self.state = insert_patient(state=self.state, patient_type=patient_type, to_block=to_block)

    def exit(self, patient):
        """
        Removes a patient from the ward.
        """
        self.inflict_cost(update_time=patient.exit_date)
        self.prev_now = self.now
        self.now = patient.exit_date
        self.state = remove_patient(state=self.state, patient_type=patient.patient_type, from_block=patient.block)
        self.patients.remove(patient)

    def inflict_cost(self, update_time):
        """
        Updates the overall cost, and returns the transofrmed reward.
        """
        resource_use = get_resource_use_per_time_unit(state=self.state)
        penalty = get_penalty_per_time_unit(state=self.state, isolation_penalty=self.isolation_penalty)
        time_since_last = update_time - self.now
        self.overall_cost += (time_since_last * (resource_use + penalty))

