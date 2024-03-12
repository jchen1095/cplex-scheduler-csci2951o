import os
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

from docplex.cp.config import context

from scheduler import Schedule, Scheduler,sudoku_example , generateVisualizerInput,solveAustraliaBinary_example
from model_timer import Timer 
import json

# Stencil created by Anirudh Narsipur February 2023
# Adapted from code written by Alexander Ding and Anirudh Narsipur


#DO NOT change this. Change run.sh to point to your local CPLEX installation
def set_context():
    solver_exec = Path(os.environ["CP_SOLVER_EXEC"])
    if not solver_exec.exists():
        print("Error: Set CP Solver Exec\n")
        exit(1)
    print(f"Using cp installation at {solver_exec}")
    context.solver.agent = "local"
    context.solver.local.execfile = str(solver_exec)



#  Poor man's Gantt chart.
#  Displays the employee schedules on the command line. 
#  Each row corresponds to a single employee. 
#  A "+" refers to a working hour and "." means no work
#  The shifts are separated with a "|"
#  The days are separated with "||"
#  This might help you analyze your solutions. 
def visualize(schedule: Schedule):
    for e in range(len(schedule)):
        print("E" + str(e + 1) + ": ", end="")
        if e < 9:
            print(" ", end="")
        for d in range(len(schedule[0])):
            for i in range(24):
                if i % 8 == 0:
                    print("|", end="")
                if (
                    schedule[e][d][0] != schedule[e][d][1]
                    and i >= schedule[e][d][0]
                    and i < schedule[e][d][1]
                ):
                    print("+", end="")
                else:
                    print(".", end="")

            print("|", end="")
        print(" ")



def main(args):
    set_context()
    # sudoku_example()
    # solveAustraliaBinary_example()
    # schedule_solver()

    input_file = Path(args.input_file)
    filename = input_file.name
   
    scheduler = Scheduler.from_file(args.input_file)
    timer = Timer()
    timer.start() 
    solution = scheduler.solve()
    timer.stop()
    
    resultdict = {}
    resultdict["Instance"] = filename
    resultdict["Time"] = round(timer.getElapsed(), 2)
    resultdict["Result"] = str(solution.n_fails)
    if solution.is_solution:
        serialized_schedule = solution #Serialize Schedule TODO get model solution
        # visualize(solution.schedule)
        # generateVisualizerInput(scheduler.config.n_employees,scheduler.config.n_days,solution.schedule) Uncomment to generate vis file
        formatted_schedule = ""
        for shift in solution.schedule:
            formatted_schedule += " ".join(map(str, shift)) + " "
        resultdict["Solution"] = formatted_schedule.strip()
        # for employee_schedule in schedule:
        # for shift_tuple in employee_schedule:
        #     formatted_schedule += f"{shift_tuple[0]} {shift_tuple[1]} "
        # formatted_schedule += "\n"
    print(json.dumps(resultdict))

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("input_file", type=str)
    args = parser.parse_args()
    main(args)
