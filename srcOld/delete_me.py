    ### Adapted definition of genetic operators
    def cxUniformModified(self, ind1, ind2):
        """
        Uniform crossover with exchange of the first gene (number of clusters)
        and the rest of the binary genes.
        """
        # Crossover of the first gene (number of clusters)

        ind1[0], ind2[0] = ind2[0], ind1[0]

        # Uniform crossover for the rest of the genes
        for i in range(1, len(ind1)):
            # Exchange occurs in the remaining genes with probability 1/2
            if random.random() < 0.5:
                ind1[i], ind2[i] = ind2[i], ind1[i]
        return ind1, ind2
    
    def mutFlipBitModified(self, ind):
        """
        Modified bit-by-bit mutation so that the number of clusters
        does not go outside the established range
        """
        mutation_type = random.choice([1,2,3])
        if mutation_type == 1:
            num_clusters_range = range(self.banda_busqueda_clusters[0],self.banda_busqueda_clusters[1])
            ind[0] = random.choice(num_clusters_range) # We choose a new number of clusters within the allowed range
        elif mutation_type == 2:
            i = random.choice(range(1, len(ind)))
            ind[i] = 1 - ind[i]  # Usual bit-flip mutation
        else:
            num_clusters_range = range(self.banda_busqueda_clusters[0],self.banda_busqueda_clusters[1])
            ind[0] = random.choice(num_clusters_range) # We choose a new number of clusters within the allowed range

            i = random.choice(range(1, len(ind)))
            ind[i] = 1 - ind[i]  # Usual bit-flip mutation
        return ind, # A tuple must always be returned by these functions
    
    def evaluate_individual(self, individual):
        return evaluate_ind(self.original_data, individual, self.eleccion_fitness, self.metodo_clust, self.linkage_hierarchical)


    def map_function(self, func, *args):
        with ProcessPoolExecutor() as executor:
            return list(executor.map(func, *args))
        
    def run(self):
        random.seed(self.seed)
        
        # Definition of the optimization problem type – we maximize chi-square
        creator.create("Fitness", base.Fitness, weights=(1.,))
        creator.create("Individual", list, fitness=creator.Fitness)

        # Initialization of the DEAP toolbox for genetic algorithms
        toolbox = base.Toolbox()

        # Creation of each gene of a chromosome
        toolbox.register("attr_bool", random.randint, 0, 1)

        # Definition of each chromosome (individual)
        toolbox.register("individual", lambda: self.init_individual(creator.Individual, toolbox.attr_bool, self.original_data))

        # The population is a list of individuals
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        # Register the evaluation function, crossover, mutation, and selection operators
        toolbox.register("evaluate", self.evaluate_individual)
        toolbox.register("mate", self.cxUniformModified)
        toolbox.register("mutate", self.mutFlipBitModified)
        toolbox.register("select", tools.selTournament, tournsize=10)

        # Parallel evaluation to speed up computations – DOES NOT WORK UNLESS python -m scoop IS USED
        toolbox.register("map", self.map_function)

        # Fitness statistics per generation
        stats_fit = tools.Statistics(lambda ind: ind.fitness.values)

        # The mean, standard deviation, minimum, and maximum fitness per generation will be shown
        stats_fit.register("avg", np.mean, axis=0)
        stats_fit.register("std", np.std, axis=0)
        stats_fit.register("min", np.min, axis=0)
        stats_fit.register("max", np.max, axis=0)

        # Logbook that will be printed on screen with the statistics    
        logbook = tools.Logbook()
        logbook.header = "gen", "nevals", "avg", "std", "min", "max"

        # init_global = time.time()
        # Creation of an initial population
        population = toolbox.population(self.npop)

        # List that will store the best individuals generation by generation
        best_inds = []
        # List that will store the average fitness of individuals in each generation
        avg_fitness_history = []
        #print(population) # check

        # Initialization of the HallOfFame
        hof = tools.HallOfFame(self.hof_size)
        
        # Initialization of the counter dictionary
        hof_counter = dict() # counts only variables as keys, not the first gene which is the number of clusters
                             # {key:(maximum fitness found for that key, number of times entered in hof)}

        # Initial evaluation of individuals
        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        # init = time.time() # for time tests
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        # fin = time.time()

        # print(f'Evaluation time of the population of generation {1}: {fin-init}') 
        # init2 = time.time()
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
            # update the dictionary with the maximum fitness for each cluster
            val=fit[0]
            cluster_number = ind[0]
            if cluster_number in self.num_clusters_and_its_max_fitness:
                if self.num_clusters_and_its_max_fitness[cluster_number][0]<val:
                    self.num_clusters_and_its_max_fitness[cluster_number]= (val,ind)
            else:
                self.num_clusters_and_its_max_fitness[cluster_number]= (val,ind)

        # fin2 = time.time()
        # print(f'Dictionary update time {1}: {fin2-init2}')
        # Update the HOF
        hof.update(population)

        # Update HOF Counter
        for ind in hof:
            key = tuple(ind[1:])
            if key in hof_counter:
                old=hof_counter[key]
                hof_counter[key] = (max(old[0],ind.fitness.values[0]),old[1]+1) # maximum fitness and count number of entries
            else :
                hof_counter[key] = (ind.fitness.values[0],1)
        
        # Definition of variables to search for convergence in the HOF
        hof_unchanged_count=0 
        latest_hof_snapshot = set(tuple(ind) for ind in hof) # Not really necessary to define it as a set since the HOF is ordered, but done for safety
        
        # Update the lists for best fitness and average fitness per generation
        if self.graficar_evolucion:
            best_inds.append(hof[0]) # Initially
            avg_fitness_history.append(np.mean([ind.fitness.values[0] for ind in population]))

        record = stats_fit.compile(population)
        #print(record)
        
        logbook.record(gen=0, nevals=len(invalid_ind), **record)
        print(logbook.stream)

        # fin_global = time.time()
        # print(f'Initialization execution time: {fin_global-init_global}')
        # Run the evolutionary algorithm
        for gen in range(self.ngen):
            print(f' \n \n --- Generation {gen+1} ---')
            #print(f'Population size: {len(population)}')
            # Select the next generation of individuals
            offspring = toolbox.select(population, len(population) - self.hof_size)

            # Mutation and reproduction of individuals
            offspring = algorithms.varAnd(offspring, toolbox, self.cxpb, self.mutpb)

            # Evaluation of individuals with invalid fitness
            # Individuals with invalid fitness are the offspring.
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            # init = time.time()
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)            
            for ind, fit in zip(invalid_ind, fitnesses):
                #print(fit)
                ind.fitness.values = fit
                # update the dictionary with the maximum fitness for each cluster
                val=fit[0]
                cluster_number = ind[0]
                if cluster_number in self.num_clusters_and_its_max_fitness:
                    if self.num_clusters_and_its_max_fitness[cluster_number][0]<val:
                        self.num_clusters_and_its_max_fitness[cluster_number]= (val,ind)
                else:
                    self.num_clusters_and_its_max_fitness[cluster_number]= (val,ind)

            # fin = time.time()

            # print(f'Evaluation time of the population of generation {gen+1}: {fin-init}') 
            # Extend the population with individuals from the hof (elitism)
            offspring.extend(hof.items)

            # Update the hall of fame with the best individuals
            hof.update(offspring)

            # Update HOF Counter
            for ind in hof:
                key = tuple(ind[1:])
                if key in hof_counter:
                    old=hof_counter[key]
                    hof_counter[key] = (max(old[0],ind.fitness.values[0]),old[1]+1) # maximum fitness and count number of entries
                else :
                    hof_counter[key] = (ind.fitness.values[0],1)
            
            # They are no longer deleted until the end
            # inds_to_delete = [key for key in hof_counter if list(key) not in [indivi[1:] for indivi in hof]]
            # for key in inds_to_delete:
            #     del hof_counter[key]

            # Show the best individual for each generation 
            print(f'Best individual of generation {gen+1} is {hof[0]} with {hof[0].fitness.values[0]}')

            if(gen==self.ngen-1): # if it is the last generation, store it
                self.ultima_gen={tuple(ind):ind.fitness.values[0] for ind in offspring}

            # Replace the population with the new generation
            population[:] = offspring

            # Update the list of best individuals and average fitness per population
            if self.graficar_evolucion:
                best_inds.append(hof[0])
                avg_fitness_history.append(np.mean([ind.fitness.values[0] for ind in population]))

            record = stats_fit.compile(population)
            #print(record)
        
            logbook.record(gen=gen+1, nevals=len(invalid_ind), **record)
            print(logbook.stream)
            
            current_hof_snapshot = set(tuple(ind) for ind in hof)
            if current_hof_snapshot == latest_hof_snapshot:
                hof_unchanged_count += 1
            else:
                hof_unchanged_count = 0
                latest_hof_snapshot = current_hof_snapshot
            
            if hof_unchanged_count >= self.convergence_generations:
                print(f'Early Stopping due to Hall Of Fame not changing in {gen+1} generations')
                self.ultima_gen={tuple(ind):ind.fitness.values[0] for ind in offspring}
                break

        # Fitness evolution over generations
        if self.graficar_evolucion:
            fitness_vals = []
            for _, ind in enumerate(best_inds, start=1):
                fitness_vals.append(ind.fitness.values[0])

            num_generations = len(fitness_vals)  
            plt.figure(figsize=(10, 5))
            plt.plot(range(1, num_generations + 1), fitness_vals, marker='o', linestyle='-', color='b', label='Best individual fitness')
            plt.plot(range(1, num_generations + 1), avg_fitness_history[:num_generations], marker='x', linestyle='--', color='r', label='Average fitness')
            plt.title('Best Individual Fitness and Average per Generation')
            plt.xlabel('Generation Number')
            plt.ylabel('Fitness')
            plt.legend()
            plt.grid(True)
            plt.show()
        self.hof=hof
        self.hof_counter=hof_counter
        return hof, hof_counter,self.num_clusters_and_its_max_fitness,self.ultima_gen
    

    def get_hof_ponderado(self): # assumes there are no control variables and uses the counter system. This is the one to use as of 21/11/2024
        var=self.numero_mejores_cromososmas_ponderacion_var_sig
        if(self.numero_mejores_cromososmas_ponderacion_var_sig==None):
            var = 2*(len(self.hof[0])-1)
        self.hof_ponderado=variable_significance_solo_dado_contador(len(self.hof[0])-1, self.hof_counter,var)
        return self.hof_ponderado
