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
            for employee in range(self.config.n_employees): #for each employee
                if(weeks == 0):
                    self.model.add(all_diff(self.shift_vars[employee][:4]))  # Assuming you want the first four days
                self.model.add((self.model.count(self.shift_vars[employee],1)) <= self.config.employee_max_total_night_shifts) #constraint for max night shifts
                hours_weekly = []
                print("Number of weeks:", self.config.n_weeks)
                for day in range(week_start,week_end): #for each day in the week collect the weekly hours
                    # shifts_weekly.append(self.shift_vars[employee][day])
                    hours_weekly.append(self.hours_vars[employee][day])
                    #self.model.add(self.hours_vars[employee][day] <= self.config.employee_max_daily) #we don't technically need
                    self.model.add(self.model.if_then((self.shift_vars[employee][day] != 0), self.hours_vars[employee][day] >= self.config.employee_min_daily))
                    self.model.add(self.model.if_then((self.shift_vars[employee][day] != 0), self.hours_vars[employee][day] <= self.config.employee_max_daily))
                    self.model.add(self.model.if_then((self.shift_vars[employee][day] == 0), self.hours_vars[employee][day] == 0))
                    if day >= 1 and day <= self.config.n_days - 1: #if the day is not the first or last day, which should account for all days
                        self.model.add(self.model.if_then((self.shift_vars[employee][day] == 1), self.shift_vars[employee][day] != self.shift_vars[employee][day - self.config.employee_max_consecutive_night_shifts]))
                        self.model.add(self.model.if_then((self.shift_vars[employee][day] == 1), self.shift_vars[employee][day] != self.shift_vars[employee][day + self.config.employee_max_consecutive_night_shifts]))     
                self.model.add(sum(hours_weekly) <= self.config.employee_max_weekly)  
                self.model.add(sum(hours_weekly) >= self.config.employee_min_weekly) 


        for day in range(self.config.n_days):
            shifts_worked = []
            hours_worked = []
            for employee in range(self.config.n_employees):
                shifts_worked.append(self.shift_vars[employee][day])
                hours_worked.append(self.hours_vars[employee][day])
            self.model.add((self.model.sum(hours_worked)) >= self.config.min_daily) #already addressed
            self.model.add((self.model.count(shifts_worked, 1)) >= self.config.min_shifts[day][1])
            self.model.add((self.model.count(shifts_worked, 2)) >= self.config.min_shifts[day][2])
            self.model.add((self.model.count(shifts_worked, 3)) >= self.config.min_shifts[day][3])
            # self.model.add((self.model.sum(hours_worked)) >= self.config.min_daily)


    def build_search(self):
      
      #TODO: Randomize day + weight
        # favored_days = []
   
        # for day in range(self.config.n_days):
        #     day_weight = 0
        #     for i in range(0,4):
        #         day_weight += self.config.min_shifts[day][i]
        #     favored_days.append((day, day_weight))
        
        # favored_days.sort(key=lambda x: x[1], reverse=True)
        # weighted_days = [[self.shift_vars[employee][day] for day, _ in favored_days] for employee in range(self.config.n_employees)]
        # # favored_shifts = [max(set(self.shift_vars[day])) for day, _ in favored_days]

        hours_worked = []
        for weeks in range(self.config.n_weeks):
            week_start = weeks * 7
            week_end = (weeks + 1) * 7
            if(weeks == 0):
                four_shift_vars = []
                for employee in range(self.config.n_employees):
                    for day in range(0,4):
                        four_shift_vars.append(self.shift_vars[employee][day])
                self.model.add(self.model.search_phase(four_shift_vars, self.model.select_random_var(), self.model.select_largest(self.model.value_impact())))
            
            

            #for day in range(self.config.n_days):

            
            #     for employee in range(self.config.n_employees):
            #        shifts_worked.append(self.shift_vars[employee][day])
            #        hours_worked.append(self.hours_vars[employee][day])
            # self.model.add(self.model.search_phase(shifts_worked, sel
                
            hours_weekly = []
            hours_today = []
            for day in range(week_start,week_end):
                shifts_worked = []
                for employee in range(self.config.n_employees):
                    hours_worked.append(self.hours_vars[employee][day])
                    hours_today.append(self.hours_vars[employee][day])
                    hours_weekly.append(self.hours_vars[employee][day])
                    shifts_worked.append(self.shift_vars[employee][day])
                self.model.add(self.model.search_phase(shifts_worked,self.model.select_random_var(),self.model.select_largest(self.model.value(), 1)))
                self.model.add(self.model.search_phase(hours_today,self.model.select_random_var(), self.model.select_smallest(self.model.value(), np.ceil(self.config.min_daily/self.config.n_employees))))
                

            self.model.add(self.model.search_phase(hours_weekly,self.model.select_random_var(), self.model.select_smallest(self.model.value(), np.ceil(self.config.min_daily/self.config.n_employees))))
                #self.model.add(self.model.search_phase(shifts_worked, self.model.select_smallest(self.model.var_index(weighted_days)), self.model.select_random_val()))
        
    #             
        self.model.add(self.model.search_phase(hours_worked, self.model.select_random_var(), self.model.select_largest(self.model.value_impact())))
      

    # def build_search(self):
        
    
    #     # shifts_worked = [] #shifts worked in total
    #     for weeks in range(self.config.n_weeks):
    #         hours_worked = []
    #         week_start = weeks * 7
    #         week_end = (weeks + 1) * 7
    #         if(weeks == 0):
    #             #four_shift_vars = [self.shift_vars[employee][:4] for employee in range(self.config.n_employees)]
    #             # four_hours_vars = [self.hours_vars[employee][:4] for employee in range(self.config.n_employees)]
    #             four_shift_vars = []
    #             for employee in range(self.config.n_employees):
    #                 for day in range(0,4):
    #                     four_shift_vars.append(self.shift_vars[employee][day])
    #                 self.model.add(self.model.search_phase(four_shift_vars, self.model.select_random_var(), self.model.select_largest(self.model.value_impact())))
    #         # for day in range(week_start,week_end):
    #         #     shifts_worked_today = [] #shifts worked per week
    #         #     hours_today = []
    #         #     for employee in range(self.config.n_employees):
    #         #         shifts_worked_today.append(self.shift_vars[employee][day])
    #         #         hours_worked.append(self.hours_vars[employee][day])
    #         #         hours_today.append(self.hours_vars[employee][day])
    #         #     #self.model.add(self.model.search_phase(shifts_worked_today, self.model.select_random_var(), self.model.select_largest(self.model.value())))
    #         #     self.model.add(self.model.search_phase(hours_today, self.model.select_random_var(), self.model.select_smallest(self.model.value(),(self.config.min_daily/self.config.n_employees))))
                
    #         hours_weekly = []
    #         for employee in range(self.config.n_employees):
    #             # shifts_worked = [] #shifts worked per week
    #             # hours_weekly = []
    #             for day in range(week_start,week_end):
    #                 hours_worked.append(self.hours_vars[employee][day])
    #                 hours_weekly.append(self.hours_vars[employee][day])
    #                 # shifts_worked.append(self.shift_vars[employee][day])
                
    #             self.model.add(self.model.search_phase(hours_weekly,self.model.select_random_var(), self.model.select_random_value()))
    #             #self.model.add(self.model.search_phase(shifts_worked, self.model.select_largest(self.model.var_local_impact()), self.model.select_largest(self.model.value())))
    #         self.model.add(self.model.search_phase(hours_worked, self.model.select_random_var(),self.model.select_random_value()))
        
            

        # shifts_worked = []
        # hours_worked = []
        # for employee in range(self.config.n_employees):
        #     for day in range(self.config.n_days):
        #         shifts_worked.append(self.shift_vars[employee][day])
        #         hours_worked.append(self.hours_vars[employee][day])
        #     self.model.add(self.model.search_phase(shifts_worked, self.model.select_random_var(), self.model.select_smallest(self.model.value())))
        #     self.model.add(self.model.search_phase(hours_worked, self.model.select_random_var(), self.model.select_largest(self.model.value_impact())))
      
                
    #     for employee in range(self.config.n_employees):
    #         self.model.add(all_diff(self.shift_vars[employee][:4]))  # Assuming you want the first four days
    #         # night_shift_count = 0 
            
    #             # print("Number of weeks:", self.config.n_weeks)
    #         self.model.add((self.model.count(self.shift_vars[employee],1)) <= self.config.employee_max_total_night_shifts)
    #         for weeks in range(self.config.n_weeks):
    #             shifts_weekly= []
    #             hours_weekly = []
    #             week_start = weeks * 7
    #             week_end = (weeks + 1) * 7
    #             for day in range(week_start, week_end):
    #                 shifts_weekly.append(self.shift_vars[employee][day])
    #                 hours_weekly.append(self.hours_vars[employee][day])
    #             self.model.add(self.model.search_phase(shifts_weekly, self.model.select_largest(self.model.var_impact()), self.model.select_largest(self.model.value_impact(),1)))
    #             self.model.search_phase(hours_weekly, self.model.select_smallest(self.model.var_success_rate()), self.model.select_largest(self.model.value_impact(),4))
    def testModel(self, schedule: [[]]) -> Boolean:

        num_days = self.config.n_days
        num_days_in_week = self.config.n_days_in_week
        employee_max_total_night_shifts = self.config.employee_max_total_night_shifts

        for employee_schedule in schedule:
            num_days_per_employee = len(employee_schedule) // 2
            if num_days_per_employee != num_days:
                print("Error: Number of days is incorrect for at least one employee.")
                return False

            if num_days_per_employee % num_days_in_week != 0:
                print("Error: Number of days is not divisible by days in a week for at least one employee.")
                return False

            num_night_shifts = sum(1 for i in range(0, len(employee_schedule), 2) if 0 <= employee_schedule[i] < employee_schedule[i + 1] <= 8)
            if num_night_shifts > employee_max_total_night_shifts:
                print(f"Error: Employee has more than {employee_max_total_night_shifts} night shifts.")
                return False
            for weeks in range(self.config.n_weeks):
                week_start = weeks * 14
                week_end = (weeks + 1) * 14
                week_hours = 0
                for day in range(week_start, week_end, 2):
                    if employee_schedule[day] != -1 and employee_schedule[day + 1] != -1:
                        week_hours += employee_schedule[day + 1] - employee_schedule[day]
                if week_hours < 20 or week_hours > 40:
                    print(f"Error: Employee has total hours outside the range [20, 40] for week {weeks} for employee {employee_schedule}.")
                    return False


        

        # one_employee = schedule[0]
        # num_days = len(one_employee)/2
        # if num_days != self.config.n_days:
        #     return False
        # if num_days % self.config.n_days_in_week != 0:
        #     print("Error: num of days is not divisible by days in week")
        #     return False
        # num_night_shifts = 0
        # for i in range(int(num_days)):
        #     day_entry = one_employee[i * 2: i * 2 + 2]
        #     if len(day_entry) != 2:
        #         print("Error: day", i, "does not have exactly two entries")
        #         return False
        #     first_entry = day_entry[0]
        #     second_entry = day_entry[1]
        #     if first_entry != -1 and second_entry != -1 and first_entry >= 0 and second_entry <=8:
        #         num_night_shifts += 1
        # if num_night_shifts > self.config.employee_max_total_night_shifts:
        #     print("Error: More than two violations of bounds (0, 8)")
        #     return False
        
        return True



    def solve(self) -> Solution:
        params = CpoParameters(
            Workers = 1,
            TimeLimit = 300,
            #Do not change the above values 
            SearchType="DepthFirst",
            LogVerbosity = "Verbose" #what is this
        )
        self.model.set_parameters(params)       
        self.build_search()
        solution = self.model.solve()
        print(solution)

        n_fails = solution.get_solver_info(CpoSolverInfos.NUMBER_OF_FAILS)
        if not solution.is_solution():
            return Solution(False, n_fails, None)
        else:
            schedule = self.construct_schedule(solution)
            if self.testModel(schedule):
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
                # if var_sol is not None:
                    # print(str(day) + " shift " + str(var_sol.get_value()))
                    # print(str(day) + " hours " + str(hour_sol.get_value()))
        
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
        

