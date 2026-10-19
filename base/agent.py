import random
import numpy as np

class Agent:
    def __init__(self, initial_cash=1.0, is_patient=True, threshold=None, signal = None):

        # caracteristicas
        self.cash = initial_cash
        self.deposited_money = 0.0
        self.bank = None                                  # Banco asignado explícitamente
        self.is_patient = is_patient                      # Tipo de preferencia de consumo
        self.signal = np.clip(signal, -1, 1)              # Señal privada s_i
        self.threshold = threshold                        # Threshold de accionar, en lugar del microfundamentado.
        self.act = False
        
        self.random_value = random.random()

        # estado
        self.has_withdrawn = False
        self.d_0 = 0

    def deposit(self, target_bank, amount=None):
        """t = 0: Asigna el banco y transfiere los fondos."""
        if amount is None:
            amount = self.cash

        if amount <= 0 or amount > self.cash:
            return

        self.cash -= amount
        self.deposited_money += amount
        self.bank = target_bank           # Guarda la identidad del banco
        self.bank.receive_deposit(self, amount)

    def decide_and_act(self):
        """
        t = 1: Decide retirar o esperar usando su banco registrado.
        """
        if self.bank is None or self.deposited_money <= 0 or self.has_withdrawn:
            return False 

        self.evaluation()

        # Impaciente siempre retira; o revisión que indica que debe actuar
        should_withdraw = (not self.is_patient) or self.act

        if should_withdraw:
            requested_amount = self.deposited_money * self.bank.r
            received_amount = self.bank.process_withdrawal(self, requested_amount)

            self.cash += received_amount
            self.deposited_money = 0.0
            self.has_withdrawn = True
        
        return should_withdraw

    def evaluation(self):
        if not self.threshold:
            # if len(self.neighbors) == 0:
            #     v = self.random_value
            # else:
            #     v = (sum(neighbor.has_withdrawn*neighbor.cash/self.bank.r for neighbor in self.neighbors))/self.d_0

            v = (sum(neighbor.has_withdrawn*neighbor.cash/self.bank.r for neighbor in self.neighbors))/self.d_0

            if v == 1:
                self.act = True
                return

            E_Y = 1 + self.bank.return_multiplier*self.signal
            E_p = (E_Y*(self.bank.i_0 - max(v*self.bank.r - self.bank.alpha,0)/self.bank.ell) + max(self.bank.alpha - v*self.bank.r,0))/(1-v)

            if E_p< self.bank.r:
                self.act = True

        else:
            if sum(neighbor.has_withdrawn for neighbor in self.neighbors) > self.threshold * len(self.neighbors):
                self.act = True
