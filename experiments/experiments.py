from timeit import default_timer as timer
import itertools
import generatebids as gb
import productmix as pm
import os
import csv
import gc
gc.disable()

def generate_file(experiment_name, n, m, M, q, rep):
    filename = ("experiments/{0}/experiment-{1}-{2}-{3}-{4}-{5}.json"
                .format(experiment_name, n, m, M, q, rep))
    print 'Generating file {}'.format(filename)
    alloc = gb.get_allocation_problem(n, m, M, q)
    pm.save_to_json(alloc, filename)

def generate_experiment_files(experiment_name, n_list, m_list, M_list, q_list, 
                                reps=10, number=1):
    # Generate folder 'experiment_name' if it does not exist
    directory = "experiments/{}".format(experiment_name)
    if not os.path.exists(directory):
            os.makedirs(directory)
    # Generate all experimental allocation files
    for n, m, M, q in itertools.product(n_list, m_list, M_list, q_list):
        for r in reps:
            generate_file(experiment_name, n, m, M, q, r)

def time_unit_min_up(filename, M, number=1):
    total_time = 0
    for _ in range(number):
        alloc = pm.load_from_json(filename)
        start = timer()
        _, steps = pm.min_up(alloc, long_step_routine="", test=True)
        end = timer()
        total_time += end-start
    avg_time = total_time / number
    return avg_time, steps

def time_binary_min_up(filename, M, number=1):
    total_time = 0
    for _ in range(number):
        alloc = pm.load_from_json(filename)
        start = timer()
        _, steps = pm.min_up(alloc, long_step_routine="binarysearch", test=True)
        end = timer()
        total_time += end-start
    avg_time = total_time / number
    return avg_time, steps

def time_demand_min_up(filename, M, number=1):
    total_time = 0
    for _ in range(number):
        alloc = pm.load_from_json(filename)
        start = timer()
        _, steps = pm.min_up(alloc, long_step_routine="demandchange", test=True)
        end = timer()
        total_time += end-start
    avg_time = total_time / number
    return avg_time, steps

def time_alloc(filename, M, number=1):
    total_time = 0
    for _ in range(number):
        alloc = pm.load_from_json(filename)
        gb.set_market_clearing_prices(alloc, M)
        start = timer()
        _, proc1, proc2, proc3 = pm.allocate(alloc, test=True)
        end = timer()
        total_time += end-start
    avg_time = total_time / number
    return avg_time, proc1, proc2, proc3

def avg(outputs):
    """Takes as input a list of output tuples and returns the element-wise
    average.
    """
    return [sum(o)/len(o) for o in zip(*outputs)]

def run_test(experiment, time_routine):
    all_outputs = []
    experiment_name, n_list, m_list, M_list, q_list, reps, number = experiment
    for n, m, M, q in itertools.product(n_list, m_list, M_list, q_list):
        out_vals = []
        for r in reps:
            filename = ("experiments/{0}/experiment-{1}-{2}-{3}-{4}-{5}.json"
                        .format(experiment_name, n, m, M, q, r))
            print "Running file {}".format(filename)
            while True:
                try:
                    out_vals.append(time_routine(filename, M, number=number)) 
                    break
                except ValueError:
                    print "Regenerating file due to ValueError"
                    generate_file(experiment_name, n, m, M, q, r)
        all_outputs.append(avg(out_vals))  # average the times for reps
    return zip(*all_outputs)

def get_data(experiment_name, f, i, j):
    filename = 'experiments/csv/{}-{}.csv'.format(experiment_name, f)
    with open(filename) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        x_coords, y_coords = [], []
        for row in reader:
            x_coords.append(row[i])
            y_coords.append(row[j])
    return x_coords, y_coords
    
def visualise(experiment_name, f, x_col, y_col):
    import matplotlib.pyplot as plt
    x_coords, y_coords = get_data(experiment_name, f, x_col, y_col)
    plt.clf()
    plt.plot(x_coords, y_coords)
    plt.ylabel('seconds')
    filename = 'experiments/figures/{}-{}.png'.format(experiment_name, experiment_type)
    plt.savefig(filename)

def save_to_file(exp_name, exp_type, *columns):
    filename = 'experiments/csv/{}-{}.csv'.format(exp_name, exp_type)
    with open(filename, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerows(zip(*columns))

def test_binary_min_up(experiment):
    return run_test(experiment, time_routine=time_binary_min_up)

def test_unit_min_up(experiment):
    return run_test(experiment, time_routine=time_unit_min_up)

def test_demand_min_up(experiment):
    return run_test(experiment, time_routine=time_demand_min_up)

def test_allocate(experiment):
    return run_test(experiment, time_routine = time_alloc)
    
def test_allocate(experiment):
    return run_test(experiment, time_routine = time_alloc)

def run_minup_tests(experiment, i):
    """Runs MinUp with unit steps, binary search and demand change techniques.
    Saves csv file with entries:
        unit_times, unit_steps,
        binary_times, binary_steps, 
        demand_times, demand_steps
    """
    print "Starting {}".format(experiment[0])
    # Run tests
    unit_times, unit_steps = test_unit_min_up(experiment)
    binary_times, binary_steps = test_binary_min_up(experiment)
    demand_times, demand_steps = test_demand_min_up(experiment)
    # Save to CSV file
    save_to_file(experiment[0], 'minup', experiment[i],
                unit_times, unit_steps,
                binary_times, binary_steps,
                demand_times, demand_steps)

def run_allocate_test(experiment, i):
    allocate_times, proc1, proc2, proc3 = test_allocate(experiment)
    # Save to CSV file
    save_to_file(experiment[0], 'allocate', experiment[i],
                allocate_times, proc1, proc2, proc3)

if __name__ == "__main__":
    ### EXPERIMENTS
    
    # control variables
    generate = False
    minup = False
    allocate = True
    plot = True

    # variables
    large_n = [10]
    small_n = [2]
    n_range = range(5, 51, 5)
    small_n_range = range(2,15)
    q_range = range(20, 501, 20)
    large_M = [10000]
    small_M = [100]
    large_q = [100]
    small_q = [50]
    m = [5]
    m_range = range(2,21)
    reps = range(0,50)
    number = 1

    ### DEFINE EXPERIMENTS
    # MinUp
    # experiment1 = ('experiment1', small_n, m, small_M, q_range, reps, number)
    # experiment2 = ('experiment2', small_n, m, large_M, q_range, reps, number)
    # experiment3 = ('experiment3', large_n, m, small_M, q_range, reps, number)
    # experiment4 = ('experiment4', large_n, m, large_M, q_range, reps, number)
    # experiment5 = ('experiment5', n_range, m, small_M, small_q, reps, number)
    # experiment6 = ('experiment6', n_range, m, large_M, small_q, reps, number)
    # experiment7 = ('experiment7', range(2,16), m, small_M, small_q, reps, number)
    # Allocate
    experiment8 = ('experiment8', small_n, m, small_M, q_range, reps, 10)
    experiment9 = ('experiment9', large_n, m, small_M, q_range, reps, 10)
    experiment10 = ('experiment10', n_range, m, small_M, small_q, reps, 2)
    experiment11 = ('experiment11', n_range, m, small_M, large_q, reps, 2)
    experiment12 = ('experiment12', small_n, m_range, small_M, small_q, reps, 10)
    experiment13 = ('experiment13', small_n, m_range, small_M, large_q, reps, 10)
    experiment14 = ('experiment14', small_n_range, m, small_M, small_q, reps, 10)
    experiment15 = ('experiment15', range(2,11), m, small_M, small_q, reps, 10)
    ### GENERATE FILES
    if generate:
        pass
        # generate_experiment_files(*experiment1)
        # generate_experiment_files(*experiment2)
        # generate_experiment_files(*experiment3)
        # generate_experiment_files(*experiment4)
        # generate_experiment_files(*experiment5)
        # generate_experiment_files(*experiment6)
        # generate_experiment_files(*experiment7)

        generate_experiment_files(*experiment8)
        # generate_experiment_files(*experiment9)
        # generate_experiment_files(*experiment10)
        # generate_experiment_files(*experiment11)
        # generate_experiment_files(*experiment12)
        # generate_experiment_files(*experiment13)
        # generate_experiment_files(*experiment14)
        # generate_experiment_files(*experiment15)

    ### MINUP EXPERIMENTS
    if minup:
        pass
        # VARY q
        # EXPERIMENT 1: MinUp vs BinaryMinUp vs DemandMinUp for small n and M
        # run_minup_tests(experiment1, 4)
        # EXPERIMENT 2: MinUp vs BinaryMinUp vs DemandMinUp for small n and large M
        # run_minup_tests(experiment2, 4)
        # EXPERIMENT 3: MinUp vs BinaryMinUp vs DemandMinUp for large n and small M
        # run_minup_tests(experiment3, 4)
        # EXPERIMENT 4: MinUp vs BinaryMinUp vs DemandMinUp for large n and large M
        # run_minup_tests(experiment4, 4)
        # VARY n
        # EXPERIMENT 5: MinUp vs BinaryMinUp vs DemandMinUp for small q and M
        # run_minup_tests(experiment5, 1)
        # EXPERIMENT 6: MinUp vs BinaryMinUp vs DemandMinUp for small q and large M
        # run_minup_tests(experiment6, 1)
        # EXPERIMENT 7: MinUp vs BinaryMinUp vs DemandMinUp for n=2..15 and small M
        # run_minup_tests(experiment7, 1)

    ### ALLOCATE EXPERIMENTS
    if allocate:
        pass
        # VARY q
        run_allocate_test(experiment8, 4)
        # run_allocate_test(experiment9, 4)
        # # VARY n
        # run_allocate_test(experiment10, 1)
        # run_allocate_test(experiment11, 1)
        # VARY m
        # run_allocate_test(experiment12, 2)
        # run_allocate_test(experiment13, 2)
        # run_allocate_test(experiment14, 1)
        # run_allocate_test(experiment15, 1)

    ### PLOT GRAPHS
    if plot:
        import matplotlib.pyplot as plt
        # plot experiments 8 and 9
        for i in [8, 9]:
            exp_name = 'experiment{}'.format(i)
            f = 'allocate'
            x_coords, y_coords = get_data(exp_name, f, 0, 1)
            fig = plt.figure(figsize=(7, 4))
            plt.clf()
            plt.scatter(x_coords, y_coords, marker="+")
            plt.ylabel('time (in seconds)')
            plt.xlabel('avg number (B) of bids per bidder')
            plt.xticks(range(0,501,100), range(0, 1251, 250))
            plt.xlim(0, 520)
            # plt.title("Experiment {}: Allocation".format(i))
            filename = 'experiments/figures/{}-{}.eps'.format(exp_name, f)
            plt.savefig(filename, format='eps')

        # plot experiments 10 and 11
        for i in [10, 11]:
            exp_name = 'experiment{}'.format(i)
            f = 'allocate'
            x_coords, y_coords = get_data(exp_name, f, 0, 1)
            fig = plt.figure(figsize=(7, 4))
            plt.clf()
            plt.scatter(x_coords, y_coords, marker="+")
            plt.ylabel('time (in seconds)')
            plt.xlabel('number of goods (n)')
            plt.xticks(range(0,51,5))
            plt.xlim(2.5, 52.5)
            # plt.title("Experiment {}: Allocation".format(i))
            filename = 'experiments/figures/{}-{}.eps'.format(exp_name, f)
            plt.savefig(filename, format='eps')
                    
        # plot experiments 12 and 13
        for i in [12, 13]:
            exp_name = 'experiment{}'.format(i)
            f = 'allocate'
            x_coords, y_coords = get_data(exp_name, f, 0, 1)
            fig = plt.figure(figsize=(7, 4))
            plt.clf()
            plt.scatter(x_coords, y_coords, marker="+")
            plt.ylabel('time (in seconds)')
            plt.xlabel('number of bidders (m)')
            plt.xticks(range(0,21,2))
            plt.xlim(1, 21)
            # plt.title("Experiment {}: Allocation".format(i))
            filename = 'experiments/figures/{}-{}.eps'.format(exp_name, f)
            plt.savefig(filename, format='eps')
        
        # plot experiments 14 (and 15)
        for i in [14]:
            exp_name = 'experiment{}'.format(i)
            f = 'allocate'
            x_coords, y_coords = get_data(exp_name, f, 0, 1)
            x_coords, y2_coords = get_data("experiment15", f, 0, 1)
            fig = plt.figure(figsize=(7, 4))
            plt.clf()
            plt.scatter(x_coords, y_coords[:len(x_coords)], marker="+")
            plt.scatter(x_coords, y2_coords, marker="x")
            plt.ylabel('time (in seconds)')
            plt.xlabel('number of goods (n)')
            plt.xticks(range(1,11))
            # plt.title("Experiment {}: Allocation".format(i))
            filename = 'experiments/figures/{}-{}.eps'.format(exp_name, f)
            plt.savefig(filename, format='eps')