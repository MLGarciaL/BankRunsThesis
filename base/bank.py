class Bank:
    def __init__(self, liquidity_ratio=0.2, liquidation_discount=0.9, return_multiplier=0.25, theta = 1, R=1.2, r = 1, equity=0):

        # caracteristicas
        self.total_deposits = 0.0
        self.alpha = liquidity_ratio      # Fracción en reservas líquidas L_0
        self.liquid_reserves = 0.0        # L_t
        self.illiquid_assets = equity     # I_t
        self.ell = liquidation_discount   # Descuento por venta forzosa (l)
        self.return_multiplier = return_multiplier
        self.theta = theta
        self.Rtilde = 0
        self.R = R
        self.r = r

        self.i_0 = 0
        
        # estados
        self.is_solvent = True
        self.depositors = {}              # Registro contable {agent_id: balance}

    def receive_deposit(self, agent_id, amount):
        self.total_deposits += amount
        self.liquid_reserves += amount * self.alpha
        self.illiquid_assets += amount * (1.0 - self.alpha)
        self.depositors[agent_id] = self.depositors.get(agent_id, 0.0) + amount 
                                    # dinero que tenia o cero          + nuevo dinero depositado 

    def process_withdrawal(self, agent_id, amount):
        if not self.is_solvent:
            return 0.0  # El banco quebró; late arrival recibe 0 bajo FCFS

        # 1. Pago directo con reservas líquidas
        if self.liquid_reserves >= amount:
            self.liquid_reserves -= amount
            self.depositors[agent_id] = 0 # self.depositors.get(agent_id, 0.0) - amount/r
            return amount

        # 2. Liquidación forzosa de activo ilíquido
        shortfall = amount - self.liquid_reserves
        required_liquidation = shortfall / self.ell

        if self.illiquid_assets >= required_liquidation:
            paid = amount
            self.liquid_reserves = 0.0
            self.illiquid_assets -= required_liquidation
            self.depositors[agent_id] = 0 # self.depositors.get(agent_id, 0.0) - paid/r
            return paid
        else:
            # 3. Quiebra: el banco entrega todo su remanente liquidado y cierra
            paid = self.liquid_reserves + (self.illiquid_assets * self.ell)
            self.liquid_reserves = 0.0
            self.illiquid_assets = 0.0
            self.is_solvent = False
            self.depositors[agent_id] = 0 #self.depositors.get(agent_id, 0.0) - paid/r
            return paid

    def settle_maturity(self):
        # Depositantes que quedan (P)
        remaining_depositors = {aid: bal for aid, bal in self.depositors.items() if bal > 0} 

        if not self.is_solvent:
            # Limpiar balance
            for agent_id, balance in remaining_depositors.items():
                self.depositors[agent_id] = 0.0
            return 

        # Retorno Bruto del banco.
        Y_theta = 1.0 + self.return_multiplier * self.theta
        self.liquid_reserves += self.illiquid_assets * Y_theta
        self.illiquid_assets = 0

        total_remaining_claims = sum(remaining_depositors.values())
        if total_remaining_claims == 0:
            return
        
        self.Rtilde = min(self.liquid_reserves/ total_remaining_claims, self.R)     
           
        # Pagar depositantes o pro-ratear si no alcanza
        
        if self.liquid_reserves >= total_remaining_claims * self.R:
            for agent_id, balance in remaining_depositors.items():
                promised_payout = balance * self.R
                agent_id.cash += promised_payout
                self.liquid_reserves -= promised_payout
                
                self.depositors[agent_id] = 0.0
        else:
            self.is_solvent = False
            L = self.liquid_reserves
            for agent_id, balance in remaining_depositors.items():
                share = balance / total_remaining_claims
                agent_id.cash += L * share
                self.liquid_reserves -= L * share
                
                self.depositors[agent_id] = 0.0

        self.liquid_reserves = 0.0 if abs(self.liquid_reserves) < 1e-10 else self.liquid_reserves
        return