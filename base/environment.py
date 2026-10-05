import numpy as np
import random
from .agent import Agent
from .bank import Bank
from .network import Network


class Environment:
    def __init__(self, 
                 N, Lambda, 
                 threshold = None,
                 liquidity_ratio=0.2, liquidation_discount=0.9, return_multiplier=0.25, theta = 1, R=1.2, r = 1, equity=0, 
                 network_type="random", p=0.2, number_of_clusters=2, p_inside=0.7, p_outside=0.01, 
                 seed=None):

        # características ambiente
        self.N = N
        self.Lambda = Lambda
        self.seed = seed
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            
        impatient_indices = set(random.sample(range(N), round(Lambda * N)))
        self.agents = [Agent(initial_cash=1.0, is_patient=(i not in impatient_indices), threshold=threshold) for i in range(N)]

        # Create bank
        self.bank = Bank(liquidity_ratio, liquidation_discount, return_multiplier, theta, R, r, equity)

        # Create network
        self.network = Network(self, network_type, p, number_of_clusters, p_inside, p_outside, seed)


        # estadísticas
        self.withdrawn_order = []
        self.withdrawn_ammount = []
        self.bank_failure = False
        self.run_percentage = 0


    def simulation(self, random_order = True, printing = True):
        for agent in self.agents:
            agent.deposit(self.bank)
            agent.signal = 1

        for agent in self.agents:
            agent.d_0 = sum(neighbor.deposited_money for neighbor in agent.neighbors) + agent.deposited_money
        
        self.bank.i_0 = self.bank.illiquid_assets/self.bank.total_deposits
        
        if printing:
            print("\033[1mt = 0\033[0m")
            print(f"Depositors: {self.N}")
            print("--- Initial balance sheet ---")
            print(f"Liquid: {round(self.bank.liquid_reserves, 2)} | " f"Illiquid: {round(self.bank.illiquid_assets, 2)} | " f"Solvent: {self.bank.is_solvent}")


        # ============================================================
        # t = 1: Sequential service
        # ============================================================

        if printing:
            print("\n\033[1mt = 1\033[0m")
            print(f"Impatient depositors: {round(self.Lambda * self.N)}")
            print(f"Patient depositors: {self.N - round(self.Lambda * self.N)}")


        # ============================================================
        # small subperiods inside t = 1, until equilibrium
        # ============================================================

        equilibrium = False
        round_number = 0

        if printing:
            print(f'    \033[1mRounds inside t = 1\033[0m')
        
        while not equilibrium:
            withdrawn_order_subperiod = [] 
            round_number += 1
            equilibrium = True
            
            if random_order:
                agents_random = random.sample(self.agents, len(self.agents))
                
            for agent in agents_random:
                changed = agent.decide_and_act()
        
                if changed:
                    withdrawn_order_subperiod.append(agent)
                    equilibrium = False

            if equilibrium == True:
                break

            self.withdrawn_order.append(withdrawn_order_subperiod)

            withdrawn_ammount_subperiod = 0
            for agent in withdrawn_order_subperiod:
                withdrawn_ammount_subperiod += agent.cash
            
            self.withdrawn_ammount.append(withdrawn_ammount_subperiod)

            if printing:
                print(f'    \033[1m1.{round_number}\033[0m')
                print(f'    Number of new withdraws: {len(self.withdrawn_order[round_number-1])}')
                

        if printing:
            print("--- After t=1 ---")
            print(f"Liquid: {round(self.bank.liquid_reserves, 2)} | " f"Illiquid: {round(self.bank.illiquid_assets, 2)} | " f"Solvent: {self.bank.is_solvent}")

        if not self.bank.is_solvent:
            self.bank_failure = True
        
        self.run_percentage = sum(agent.has_withdrawn for agent in self.agents)/len(self.agents)
            
        
        # ============================================================
        # t = 2: ?
        # ============================================================

        if printing:
            print("\n\033[1mt = 2\033[0m")
        
        self.bank.settle_maturity()

        if printing:
            print("--- Maturity settlement ---")
            print(f"Liquid: {round(self.bank.liquid_reserves, 2)} | " f"Illiquid: {round(self.bank.illiquid_assets, 2)} | " f"Solvent: {self.bank.is_solvent}")
        
        # ============================================================
        # Payoffs
        # ============================================================

        if printing:
            print("\n--- Individual Payoffs ---")
        
            for i, agent in enumerate(self.agents):
                depositor_type = "Patient" if agent.is_patient else "Impatient"
                print(f"Agent {i}: " f"{depositor_type} | " f"Payoff = {round(agent.cash, 2)}")

    def graph(self):
        self.network.graph()