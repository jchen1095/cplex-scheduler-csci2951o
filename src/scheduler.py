from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from docplex.cp.config import *
from docplex.cp.model import *
from cpinstance import CPInstance
# silence logs
context.set_attribute("log_output", None)


def sudoku_example():
    def get_box(grid, i):
    #get the i'th box
        box_row = (i // 3) * 3
        box_col = (i % 3) * 3
        box = []
        for row in range(box_row, box_row + 3):
            for col in range(box_col, box_col + 3):
                box.append(grid[row][col])
        return box

    model = CpoModel()
    int_vars = np.array([np.array([integer_var(1,9) for j in range(0,9)]) for i in range(0,9)])
    #Columns are different
    for row in int_vars:
        model.add(all_diff(row.tolist()))
    #Rows are different
    for col_index in range(0,9):
        model.add(all_diff(int_vars[:,col_index].tolist()))
    
    for box in range(0,9):
        model.add(all_diff(get_box(int_vars,box)))
    sol = model.solve()
    print(sol)
    print(sol[int_vars[1,1]])
    if not sol.is_solution():
        print("ERROR")
    else:
        for i in range(0,9):
            for j in range(0,9):
                print(sol[int_vars[i,j]],end=" ")
            print()
    

def solveAustraliaBinary_example():
    Colors = ["red", "green", "blue"]
    try: 
        cp = CpoModel() 
        
        WesternAustralia =  integer_var(0,3)
        NorthernTerritory = integer_var(0,3)
        SouthAustralia = integer_var(0,3)
        Queensland = integer_var(0,3)
        NewSouthWales = integer_var(0,3)
        Victoria = integer_var(0,3)
        
        cp.add(WesternAustralia != NorthernTerritory)
        cp.add(WesternAustralia != SouthAustralia)
        cp.add(NorthernTerritory != SouthAustralia)
        cp.add(NorthernTerritory != Queensland)
        cp.add(SouthAustralia != Queensland)
        cp.add(SouthAustralia != NewSouthWales)
        cp.add(SouthAustralia != Victoria)
        cp.add(Queensland != NewSouthWales)
        cp.add(NewSouthWales != Victoria)

        params = CpoParameters(
            Workers = 1,
            TimeLimit = 300,
            SearchType="DepthFirst" 
        )
        cp.set_parameters(params)
        
        sol = cp.solve() 
        print(sol)
        if sol.is_solution(): 
            
            print( "\nWesternAustralia:    " + Colors[sol[WesternAustralia]])
            print( "NorthernTerritory:   " +   Colors[sol[NorthernTerritory]])
            print( "SouthAustralia:      " +   Colors[sol[SouthAustralia]])
            print( "Queensland:          " +   Colors[sol[Queensland]])
            print( "NewSouthWales:       " +   Colors[sol[NewSouthWales]])
            print( "Victoria:            " +   Colors[sol[Victoria]])
        else:
            print("No Solution found!");
        
    except Exception as e:
        print(f"Error: {e}")

    

# [Employee][Days][startTime][EndTime]
Schedule = list[list[tuple[int, int]]]

@dataclass
class Solution:
    is_solution: bool #Is this a solution
    n_fails: int # Number of failures reported by the model
    schedule: Optional[Schedule] #The Employee Schedule. Should not be None if is_solution is true


class Scheduler:
    OFF_SHIFT = 0
    NIGHT_SHIFT = 1

    def __init__(self, config: CPInstance):
        self.config = config
        self.model = CpoModel()
        self.shift_vars = [[integer_var(0, 3) for _ in range(self.config.n_days + 1)] for _ in range(self.config.n_employees + 1)]
        self.hours_vars = [[integer_var(0, 8) for _ in range(self.config.n_days + 1)] for _ in range(self.config.n_employees + 1)]
        self.build_constraints()
        

    def build_constraints(self):
        #need to add constraint for weeks

        for weeks in range(self.config.n_weeks):
            print("week:", weeks)
            week_start = weeks * 7
            week_end = (weeks + 1) * 7
            print(f"Week {weeks}: Days {week_start} to {week_end}")
            for employee in range(self.config.n_employees):
                self.model.add(all_diff(self.shift_vars[employee][:4]))  # Assuming you want the first four days
            # night_shift_count = 0 
            
                print("Number of weeks:", self.config.n_weeks)
                self.model.add(sum(self.hours_vars[employee][(week_start):week_end]) <= self.config.employee_max_weekly)  
                self.model.add(sum(self.hours_vars[employee][(week_start):week_end]) >= self.config.employee_min_weekly) 
    
                self.model.add((self.model.count(self.shift_vars[employee],1)) <= self.config.employee_max_total_night_shifts)
                for day in range(self.config.n_days):
                    # self.model.add((self.model.count(self.shift_vars[:, day],0),self.config.min_shifts[day][0],self.config.n_employees))
                    # self.model.add((self.model.sum(self.hours_vars[:][day]) >= self.config.min_daily))
                    # self.model.add((self.model.sum(self.hours_vars[day]) >= self.config.min_daily))
                    self.model.add(self.model.if_then((self.shift_vars[employee][day] != 0), self.hours_vars[employee][day] >= 4))
                    self.model.add(self.hours_vars[employee][day] <= self.config.employee_max_daily)
                    self.model.add(self.model.if_then((self.shift_vars[employee][day] == 0), self.hours_vars[employee][day] == 0))
                    if day >= 1 and day <= self.config.n_days - 1: #if the day is not the first or last day, which should account for all days
                        self.model.add(self.model.if_then((self.shift_vars[employee][day] == 1), self.shift_vars[employee][day] != self.shift_vars[employee][day - self.config.employee_max_consecutive_night_shifts]))
                        self.model.add(self.model.if_then((self.shift_vars[employee][day] == 1), self.shift_vars[employee][day] != self.shift_vars[employee][day + self.config.employee_max_consecutive_night_shifts]))
                    # if self.model.eq(self.shift_vars[employee][day], 3):
                    #     night_shift_count += 1
            

        for day in range(self.config.n_days):
            shifts_worked = []
            hours_worked = []
            for employee in range(self.config.n_employees):
                shifts_worked.append(self.shift_vars[employee][day])
                hours_worked.append(self.hours_vars[employee][day])
            self.model.add((self.model.count(shifts_worked, 1)) >= self.config.min_shifts[day][1])
            self.model.add((self.model.count(shifts_worked, 0)) >= self.config.min_shifts[day][0])
            self.model.add((self.model.count(shifts_worked, 2)) >= self.config.min_shifts[day][2])
            self.model.add((self.model.count(shifts_worked, 3)) >= self.config.min_shifts[day][3])
            # print(self.config.min_daily)
            # print(self.model.sum(hours_worked))
            self.model.add((self.model.sum(hours_worked)) >= self.config.min_daily)
            
                

                
            
        
                
    
    # self.model.add(sum(self.shift_vars) >= self.config.min_shifts[day][shift])
        

    def solve(self) -> Solution:
        params = CpoParameters(
            Workers = 1,
            TimeLimit = 300,
            #Do not change the above values 
            # SearchType="DepthFirst" Uncomment for part 2
            # LogVerbosity = "Verbose"
        )
        self.model.set_parameters(params)       
        

        solution = self.model.solve()
        print(solution)

        n_fails = solution.get_solver_info(CpoSolverInfos.NUMBER_OF_FAILS)
        if not solution.is_solution():
            return Solution(False, n_fails, None)
        else:
            schedule = self.construct_schedule(solution)
            return Solution(True, n_fails, schedule)

    def construct_schedule(self, solution: CpoSolveResult) -> Schedule:
        """Convert the solution as reported by the model
        to an employee schedule (see handout) that can be returned

        Args:
            solution (CpoSolveResult): The solution as returned by the model

        Returns:
            Schedule: An output schedule that can returned
            NOTE: Schedule must be in format [Employee][Days][startTime][EndTime]
        """
        schedule = [[] for _ in range(self.config.n_employees)]

        for employee in range(0, self.config.n_employees):
            for day in range(0, self.config.n_days):
                shift_type = solution.get_var_solution(self.shift_vars[employee][day])
                hours_worked = solution.get_var_solution(self.hours_vars[employee][day])
                if shift_type.get_value() == 0 and hours_worked.get_value() == 0:
                    schedule[employee].extend([-1, -1])
                if shift_type.get_value() == 1:
                    schedule[employee].extend([0, 0 + hours_worked.get_value()])
                if shift_type.get_value() == 2:
                    schedule[employee].extend([8, 8 + hours_worked.get_value()])
                if shift_type.get_value() == 3:
                    schedule[employee].extend([16, 16 + hours_worked.get_value()])
                var_sol = solution.get_var_solution(self.shift_vars[employee][day])
                hour_sol = solution.get_var_solution(self.hours_vars[employee][day])
                if var_sol is not None:
                    print(str(day) + " shift " + str(var_sol.get_value()))
                    print(str(day) + " hours " + str(hour_sol.get_value()))
        
        for i, employee_schedule in enumerate(schedule):
            print(f"Employee {i}: {employee_schedule}")
        return schedule

    @staticmethod
    def from_file(f) -> Scheduler:
        # Create a scheduler instance from a config file
        config = CPInstance.load(f)
        return Scheduler(config)

'''
   * Generate Visualizer Input
   * author: Adapted from code written by Lily Mayo
   *
   * Generates an input solution file for the visualizer. 
   * The file name is numDays_numEmployees_sol.txt
   * The file will be overwritten if it already exists.
   * 
   * @param numEmployees the number of employees
   * @param numDays the number of days
   * @param beginED int[e][d] the hour employee e begins work on day d, -1 if not working
   * @param endED   int[e][d] the hour employee e ends work on day d, -1 if not working
   '''
def generateVisualizerInput(numEmployees : int, numDays :int,  sched : Schedule ):
    solString = f"{numDays} {numEmployees}\n"

    for d in range(0,numDays):
        for e in range(0,numEmployees):
            solString += f"{sched[e][d][0]} {sched[e][d][1]}\n"

    fileName = f"{str(numDays)}_{str(numEmployees)}_sol.txt"

    try:
        with open(fileName,"w") as fl:
            fl.write(solString)
        fl.close()
    except IOError as e:
        print(f"An error occured: {e}")
        

