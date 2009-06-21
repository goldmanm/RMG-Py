#!/usr/bin/python
# -*- coding: utf-8 -*-

################################################################################
#
#	RMG - Reaction Mechanism Generator
#
#	Copyright (c) 2002-2009 Prof. William H. Green (whgreen@mit.edu) and the
#	RMG Team (rmg_dev@mit.edu)
#
#	Permission is hereby granted, free of charge, to any person obtaining a
#	copy of this software and associated documentation files (the 'Software'),
#	to deal in the Software without restriction, including without limitation
#	the rights to use, copy, modify, merge, publish, distribute, sublicense,
#	and/or sell copies of the Software, and to permit persons to whom the
#	Software is furnished to do so, subject to the following conditions:
#
#	The above copyright notice and this permission notice shall be included in
#	all copies or substantial portions of the Software.
#
#	THE SOFTWARE IS PROVIDED 'AS IS', WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#	IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#	FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#	AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#	LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
#	FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
#	DEALINGS IN THE SOFTWARE.
#
################################################################################

"""
Contains classes for working with the reaction model generated by RMG.
"""

import logging

import kinetics

################################################################################

class ReactionModel:
	"""
	Represent a generic reaction model. A reaction model consists of `species`,
	a list of species, and `reactions`, a list of reactions.
	"""	

	def __init__(self, species=None, reactions=None):
		self.species = species or []
		self.reactions = reactions or []

################################################################################

class CoreEdgeReactionModel:
	"""
	Represent a reaction model constructed using a rate-based screening
	algorithm. The `core` is a reaction model that represents species and 
	reactions currently in the model, while the `edge` is a reaction model that
	represents species and reactions identified as candidates for addition to 
	the core.
	"""	

	def __init__(self, core=None, edge=None):
		if core is None:
			self.core = ReactionModel()
		else:
			self.core = core
		if edge is None:
			self.edge = ReactionModel()
		else:
			self.edge = edge

	def initialize(self, coreSpecies):
		"""
		Initialize a reaction model with a list `coreSpecies` of species to
		start out with.
		"""

		logging.info('')
		
		for species1 in coreSpecies:
			# Generate reactions if reactive
			rxnList = []
			if species1.reactive:
				# Generate unimolecular reactions
				rxnList.extend(kinetics.database.getReactions([species1]))
				# Generate bimolecular reactions
				for species2 in self.core.species:
					rxnList.extend(kinetics.database.getReactions([species1, species2]))
			# Add to core
			self.addSpeciesToCore(species1)
			# Add to edge
			for rxn in rxnList:
				for spec in rxn.products:
					if spec not in self.edge.species:
						self.addSpeciesToEdge(spec)
				self.addReactionToEdge(rxn)

		logging.info('')
		logging.info('After core-edge reaction model initialization:')
		logging.info('\tThe model core has %s species and %s reactions' % (len(self.core.species), len(self.core.reactions)))
		logging.info('\tThe model edge has %s species and %s reactions' % (len(self.edge.species), len(self.edge.reactions)))

		# We cannot conduct simulations without having at least one reaction
		# in the core because otherwise we have no basis for selecting the
		# characteristic flux needed to test for model validity; thus we must
		# enlarge the reaction model until at least one reaction is in the core
		#while len(self.core.reactions) == 0:
		#	self.enlarge()

	def addSpeciesToCore(self, spec):
		"""
		Add a species `spec` to the reaction model core (and remove from edge if
		necessary). This function also moves any reactions in the edge that gain
		core status as a result of this change in status to the core.
		"""

		# Add the species to the core
		self.core.species.append(spec)

		if spec in self.edge.species:

			# If species was in edge, remove it
			self.edge.species.remove(spec)

			# Search edge for reactions that now contain only core species;
			# these belong in the model core and will be moved there
			rxnList = []
			for rxn in self.edge.reactions:
				allCore = True
				for reactant in rxn.reactants:
					if reactant not in self.core.species: allCore = False
				for product in rxn.products:
					if product not in self.core.species: allCore = False
				if allCore: rxnList.append(rxn)

			# Move any identified reactions to the core
			for rxn in rxnList:
				self.addReactionToCore(rxn)


	def addSpeciesToEdge(self, spec):
		"""
		Add a species `spec` to the reaction model edge.
		"""
		self.edge.species.append(spec)

	def addReactionToCore(self, rxn):
		"""
		Add a reaction `rxn` to the reaction model core (and remove from edge if
		necessary). This function assumes `rxn` has already been checked to
		ensure it is supposed to be a core reaction (i.e. all of its reactants
		AND all of its products are in the list of core species).
		"""
		self.core.reactions.append(rxn)
		if rxn in self.edge.reactions:
			self.edge.reactions.remove(rxn)

	def addReactionToEdge(self, rxn):
		"""
		Add a reaction `rxn` to the reaction model edge. This function assumes
		`rxn` has already been checked to ensure it is supposed to be an edge
		reaction (i.e. all of its reactants OR all of its products are in the
		list of core species, and the others are in either the core or the
		edge).
		"""
		self.edge.reactions.append(rxn)

################################################################################

class TemperatureModel:
	"""
	Represent a temperature profile. Currently the only implemented model is
	isothermal (constant temperature).
	"""

	def __init__(self):
		self.type = ''
		self.temperatures = []
		
	def isIsothermal(self):
		return self.type == 'isothermal'
	
	def setIsothermal(self, temperature):
		self.type = 'isothermal'
		self.temperatures = [ [0.0, temperature] ]
	
	def getTemperature(self, time):
		if self.isIsothermal():
			return self.temperatures[0][1]
		else:
			return None
	
	def __str__(self):
		string = 'Temperature model: ' + self.type + ' '
		if self.isIsothermal():
			string += str(self.getTemperature(0))
		return string
	
################################################################################

class PressureModel:
	"""
	Represent a pressure profile. Currently the only implemented model is
	isobaric (constant pressure).
	"""
	
	def __init__(self):
		self.type = ''
		self.pressures = []
		
	def isIsobaric(self):
		return self.type == 'isobaric'
	
	def setIsobaric(self, pressure):
		self.type = 'isobaric'
		self.pressures = [ [0.0, pressure] ]
	
	def getPressure(self, time):
		if self.isIsobaric():
			return self.pressures[0][1]
		else:
			return None

	def __str__(self):
		string = 'Pressure model: ' + self.type + ' '
		if self.isIsobaric():
			string += str(self.getPressure(0))
		return string

################################################################################

class ReactionSystem:
	"""
	Represent a reaction system. A reaction system is defined by a temperature 
	model *temperatureModel*, a pressure model *pressureModel*, and a 
	dictionary of initial and constant concentrations *initialConcentration*.
	
	Each RMG job can handle multiple reaction systems; the resulting model
	will generally be the union of the models that would have been generated 
	via individual RMG jobs, and will therefore be valid across all reaction
	systems provided.	
	"""

	def __init__(self, temperatureModel=None, pressureModel=None, \
	             initialConcentration={}):
		if temperatureModel is None:
			self.temperatureModel = TemperatureModel()
		else:
			self.temperatureModel = temperatureModel
		if pressureModel is None:
			self.pressureModel = PressureModel()
		else:
			self.pressureModel = pressureModel
		self.initialConcentration = initialConcentration
	
	