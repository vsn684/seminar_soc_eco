import agentpy as ap
import numpy as np

# Funktion til beregning af Gini-koefficient
def gini(x):
    """Beregner Gini-koefficienten."""
    x = np.array(x)
    mean_x = x.mean()
    mad = np.abs(np.subtract.outer(x, x)).mean()
    gini_coefficient = 0.5 * mad / mean_x
    return gini_coefficient

# Agent-definition for arbejdere uden arbejdsløshedsforsikring (UI)
class Worker(ap.Agent):
    def setup(self):
        self.status = False
        self.separation_week = None  # Registrer hvornår arbejderen blev arbejdsløs
        self.wealth = self.model.p.get('initial_wealth', 0)  # Standardværdi 0 hvis ikke angivet

    def gain_wealth(self):
        if self.status:
            self.wealth += 2

# Agent-definition for arbejdere med arbejdsløshedsforsikring (UI)
class WorkerUI(ap.Agent):
    def setup(self):
        self.status = False
        self.separation_week = None  # Registrer hvornår arbejderen blev arbejdsløs
        self.initially_employed = False  # Initialiser til False for alle agenter
        self.eligibility = 0  # Uger med ret til arbejdsløshedsdagpenge tilbage
        self.wealth = self.model.p.get('initial_wealth', 0)  # Standardværdi 0 hvis ikke angivet
        self.benefit_weeks = 0  # Antal uger med modtagelse af dagpenge
        self.initial_eligibility = 0  # Berettigelse på tidspunktet for arbejdsløshed
        self.skill_level = np.random.randint(1, 11)  # Færdighedsniveau mellem 1 og 10

    def gain_wealth(self):
        if self.status:  # Ansat
            if self.model.almp:  # Inkluder kun færdighedsbaseret indkomstforøgelse hvis ALMP er aktiveret
                self.wealth += 2 + self.model.skill_income_factor * self.skill_level
                self.skill_level += 1
            else:
                self.wealth += 2  # Standard indkomstforøgelse uden færdighedspåvirkning
            self.benefit_weeks = 0  # Nulstil uger med ydelser når ansat
        elif self.eligibility > 0:  # Modtager arbejdsløshedsdagpenge
            self.wealth += 0.5  # Øg formue pga. dagpenge
            self.benefit_weeks += 1  # Øg antal uger med ydelser
            if self.model.almp and self.benefit_weeks >= 3:  # Forøg færdighed kun hvis ALMP er aktiveret
                self.skill_level += 0.5

# Modeldefinition for arbejdsmarked uden UI
class LaborMarketModel(ap.Model):
    def setup(self):
        np.random.seed(self.p['seed'])
        self.labor_force = self.p['labor_force']
        self.separation_rate = self.p['separation_rate']
        self.initial = self.p['initial']
        self.target = self.p['target']
        self.hiring_factor = self.p.get('hiring_factor', 1.0)  # Standardværdi 1.0 hvis ikke angivet
        self.steps = self.p.get('steps', 70)
        self.workers = ap.AgentList(self, self.labor_force, Worker)

        employed_workers = self.workers.random(int(self.labor_force * self.initial))
        for worker in employed_workers:
            worker.status = True

        # Sæt eksogene makrodata (ledige stillinger)
        self.vacancies = []
        for week in range(1, 20):
            self.vacancies.append(self.labor_force * self.initial)
        for week in range(20, 30):
            self.vacancies.append(
                self.vacancies[-1] - ((self.initial - self.target) / 11) * self.labor_force
            )
        for week in range(30, 41):
            self.vacancies.append(self.target * self.labor_force)
        for week in range(41, 51):
            self.vacancies.append(
                self.vacancies[-1] + ((self.initial - self.target) / 11) * self.labor_force
            )
        for week in range(51, self.steps + 1):
            self.vacancies.append(self.labor_force * self.initial)

    def calculate_unemployment_rate(self):
        unemployed = len([w for w in self.workers if not w.status])
        return (unemployed / self.labor_force) * 100

    def calculate_gini_coefficient(self):
        wealth_distribution = [w.wealth for w in self.workers]
        return gini(wealth_distribution)

    def step(self):
        current_week = self.t
        unemployed_workers = [w for w in self.workers if not w.status]
        employed_workers = [w for w in self.workers if w.status]

        LEAVS = self.separation_rate * len(employed_workers)
        DIFEMP = (
            self.vacancies[current_week - 1] - self.vacancies[current_week - 2]
            if current_week > 1
            else 0
        )

        DIFEMP = int(DIFEMP * self.hiring_factor)

        separations = int(LEAVS - DIFEMP) if DIFEMP < 0 else int(LEAVS)
        if len(employed_workers) >= separations and separations > 0:
            newly_unemployed = np.random.choice(employed_workers, separations, replace=False)
            for worker in newly_unemployed:
                worker.status = False
                worker.separation_week = current_week

        unemployed_workers = [w for w in self.workers if not w.status]
        employed_workers = [w for w in self.workers if w.status]

        hires = int(LEAVS + DIFEMP) if DIFEMP > 0 else int(LEAVS)
        available_workers = [w for w in unemployed_workers if w.separation_week != current_week]
        hired_workers = np.random.choice(
            available_workers, min(hires, len(available_workers)), replace=False
        )

        for worker in hired_workers:
            worker.status = True
            worker.separation_week = None

        for worker in employed_workers:
            worker.gain_wealth()

        self.record('unemployment_rate', self.calculate_unemployment_rate())
        self.record('gini_coefficient', self.calculate_gini_coefficient())
        self.record('t', self.t)

    def end(self):
        pass  # Plotning håndteres uden for modellen

# Modeldefinition for arbejdsmarked med UI og hiring_factor
class LaborMarketModelUI(ap.Model):
    def setup(self):
        np.random.seed(self.p['seed'])
        self.labor_force = self.p['labor_force']
        self.separation_rate = self.p['separation_rate']
        self.initial = self.p['initial']
        self.target = self.p['target']
        self.hiring_factor = self.p.get('hiring_factor', 1.0)
        self.steps = self.p.get('steps', 70)
        self.almp = self.p.get('almp', False)
        self.skill_income_factor = self.p.get('skill_income_factor', 0.03)  # Standardværdi 0.03

        self.rolling_initial_eligibility = self.p.get('rolling_initial_eligibility', False)
        self.workers = ap.AgentList(self, self.labor_force, WorkerUI)
        np.random.seed(self.p['seed'])

        if not self.rolling_initial_eligibility:
            employed_workers = self.workers.random(int(self.labor_force * self.initial))
            for worker in employed_workers:
                worker.status = True
                worker.initially_employed = True  # Marker som oprindeligt ansat
                worker.eligibility = 6

            unemployed_workers = [w for w in self.workers if not w.status]
            EL = 0
            for worker in unemployed_workers:
                worker.status = False
                worker.separation_week = None
                worker.eligibility = EL
                EL = (EL + 1) % 7  # Berettigelse varierer fra 0 til 6
        else:
            EL = 0
            for worker in self.workers:
                worker.eligibility = EL
                EL = (EL + 1) % 7  # Berettigelse varierer fra 0 til 6

            employed_workers = self.workers.random(int(self.labor_force * self.initial))
            for worker in self.workers:
                if worker in employed_workers:
                    worker.status = True
                    worker.separation_week = None
                else:
                    worker.status = False
                    worker.separation_week = None

        # Sæt eksogene makrodata (ledige stillinger)
        self.vacancies = []
        for week in range(1, 20):
            self.vacancies.append(self.labor_force * self.initial)
        for week in range(20, 30):
            self.vacancies.append(
                self.vacancies[-1] - ((self.initial - self.target) / 11) * self.labor_force
            )
        for week in range(30, 41):
            self.vacancies.append(self.target * self.labor_force)
        for week in range(41, 51):
            self.vacancies.append(
                self.vacancies[-1] + ((self.initial - self.target) / 11) * self.labor_force
            )
        for week in range(51, self.steps + 1):
            self.vacancies.append(self.labor_force * self.initial)

        self.vacancies_carried_over = 0

    def calculate_unemployment_rate(self):
        unemployed = len([w for w in self.workers if not w.status])
        return (unemployed / self.labor_force) * 100

    def calculate_gini_coefficient(self):
        wealth_distribution = [w.wealth for w in self.workers]
        return gini(wealth_distribution)

    def step(self):
        current_week = self.t
        unemployed_workers = [w for w in self.workers if not w.status]
        employed_workers = [w for w in self.workers if w.status]

        LEAVS = self.separation_rate * len(employed_workers)
        DIFEMP = (
            self.vacancies[current_week - 1] - self.vacancies[current_week - 2]
            if current_week > 1
            else 0
        )
        DIFEMP = int(DIFEMP * self.hiring_factor)

        separations = int(LEAVS - DIFEMP) if DIFEMP < 0 else int(LEAVS)
        if len(employed_workers) >= separations and separations > 0:
            # Normaliser sandsynligheder baseret på færdighedsniveau eller andre kriterier
            total_skill = sum(worker.skill_level for worker in employed_workers)
            if total_skill > 0:
                probabilities = [worker.skill_level / total_skill for worker in employed_workers]
            else:
                probabilities = [1 / len(employed_workers)] * len(employed_workers)

            # Vælg arbejdere til afskedigelse probabilistisk
            newly_unemployed = np.random.choice(
                employed_workers, separations, replace=False, p=probabilities
            )
            for worker in newly_unemployed:
                worker.status = False
                worker.separation_week = current_week
                worker.eligibility = min(worker.eligibility, 6)
                worker.initial_eligibility = worker.eligibility  # Gem berettigelse ved arbejdsløshed

        unemployed_workers = [w for w in self.workers if not w.status]
        employed_workers = [w for w in self.workers if w.status]

        hires = int(LEAVS + DIFEMP) if DIFEMP > 0 else int(LEAVS)
        hires += self.vacancies_carried_over
        self.vacancies_carried_over = 0

        vacancies_left = 0

        if self.almp:
            # ALMP aktiveret: Brug sandsynlighedsbaseret udvælgelse
            total_skill = sum(w.skill_level for w in unemployed_workers)
            threat_factor = self.p.get('threat_factor', 1)

            def calculate_probabilities():
                if total_skill > 0:
                    raw_probabilities = [
                        w.skill_level / total_skill
                        for w in unemployed_workers
                    ]
                    prob_sum = sum(raw_probabilities)
                    return [p / prob_sum for p in raw_probabilities] if prob_sum > 0 else None
                else:
                    return None

            probabilities = calculate_probabilities()

            for _ in range(hires):
                NFAIL = 0
                while NFAIL < 10:
                    if not unemployed_workers:
                        break

                    if probabilities:
                        candidate = np.random.choice(unemployed_workers, p=probabilities)
                    else:
                        break  # Ingen gyldige sandsynligheder, afslut ansættelsesloop

                    if candidate.separation_week == current_week:
                        NFAIL += 1
                        continue

                    # Kandidater i deres sidste uges berettigelse accepterer straks
                    if candidate.eligibility <= 1:
                        candidate.status = True
                        candidate.separation_week = None
                        unemployed_workers.remove(candidate)  # Fjern kandidat efter ansættelse
                        total_skill = sum(w.skill_level for w in unemployed_workers)
                        probabilities = calculate_probabilities()  # Beregn sandsynligheder igen
                        break  # Stilling besat, gå til næste stilling

                    # Kandidater i locking-in perioden afviser tilbud
                    elif candidate.benefit_weeks >= 3:
                        NFAIL += 0
                        continue

                    # Kandidater i deres første to uger med ydelser accepterer med 5% sandsynlighed
                    elif candidate.benefit_weeks < 3 and np.random.random() <= 0.05:
                        candidate.status = True
                        candidate.separation_week = None
                        unemployed_workers.remove(candidate)  # Fjern kandidat efter ansættelse
                        total_skill = sum(w.skill_level for w in unemployed_workers)
                        probabilities = calculate_probabilities()  # Beregn sandsynligheder igen
                        break  # Stilling besat, gå til næste stilling

                    # Standardafvisning for alle andre tilfælde
                    else:
                        NFAIL += 1
                        continue

                else:
                    vacancies_left += 1

        else:
            # ALMP deaktiveret: Brug normal tilfældig udvælgelse
            for _ in range(hires):
                NFAIL = 0
                while NFAIL < 10:
                    if not unemployed_workers:
                        break
                    candidate = np.random.choice(unemployed_workers)
                    if candidate.separation_week == current_week:
                        NFAIL += 1
                        continue
                    if candidate.eligibility >= 2:
                        NFAIL += 1
                        continue
                    else:
                        candidate.status = True
                        candidate.separation_week = None
                        unemployed_workers.remove(candidate)
                        break
                else:
                    vacancies_left += 1

        # Opdater overførte ledige stillinger
        self.vacancies_carried_over += vacancies_left

        # Opdater berettigelse for alle arbejdere
        for worker in self.workers:
            if not worker.status:
                worker.eligibility = max(worker.eligibility - 1, 0)
            else:
                worker.eligibility = min(worker.eligibility + 1 / 3, 6)

        for worker in employed_workers:
            worker.gain_wealth()

        for worker in unemployed_workers:
            if worker.eligibility > 0:
                worker.gain_wealth()

        self.record('unemployment_rate', self.calculate_unemployment_rate())
        self.record('gini_coefficient', self.calculate_gini_coefficient())
        self.record('t', self.t)

    def end(self):
        pass