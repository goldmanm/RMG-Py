#!/usr/bin/python
# -*- coding: utf-8 -*-

################################################################################
#
#   RMG - Reaction Mechanism Generator
#
#   Copyright (c) 2002-2010 Prof. William H. Green (whgreen@mit.edu) and the
#   RMG Team (rmg_dev@mit.edu)
#
#   Permission is hereby granted, free of charge, to any person obtaining a
#   copy of this software and associated documentation files (the 'Software'),
#   to deal in the Software without restriction, including without limitation
#   the rights to use, copy, modify, merge, publish, distribute, sublicense,
#   and/or sell copies of the Software, and to permit persons to whom the
#   Software is furnished to do so, subject to the following conditions:
#
#   The above copyright notice and this permission notice shall be included in
#   all copies or substantial portions of the Software.
#
#   THE SOFTWARE IS PROVIDED 'AS IS', WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
#   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
#   DEALINGS IN THE SOFTWARE.
#
################################################################################

"""
This module contains functionality for generating mechanisms with isotopes.
"""

import os
import os.path
import logging
import numpy as np
import itertools
from copy import copy
import pandas as pd
import shutil
import math

import rmgpy.constants as constants
from rmgpy.molecule import Molecule
from rmgpy.molecule.element import getElement
from rmgpy.tools.loader import loadRMGJob
from rmgpy.chemkin import ChemkinWriter
from rmgpy.rmg.input import getInput
from rmgpy.rmg.main import RMG, initializeLog
from rmgpy.species import Species
from rmgpy.reaction import Reaction
from rmgpy.data.kinetics.family import TemplateReaction
from rmgpy.rmg.listener import SimulationProfileWriter
from rmgpy.thermo.thermoengine import processThermoData
from rmgpy.data.thermo import findCp0andCpInf

def initializeIsotopeModel(rmg, isotopes):
    """
    Initialize the RMG object by using the parameter species list
    as initial species instead of the species from the RMG input file.

    """
    # Read input file
    rmg.loadInput(rmg.inputFile)

    # Check input file 
    rmg.checkInput()

    # Load databases
    rmg.loadDatabase()

    logging.info("isotope: Adding the isotopomers into the RMG model")
    for spc in isotopes:
        spec, isNew = rmg.reactionModel.makeNewSpecies(spc)
        spec.thermo = spc.thermo        
        if isNew:
            rmg.reactionModel.addSpeciesToEdge(spec)
            rmg.initialSpecies.append(spec)
    logging.info("isotope: Adding standard species into the model")
    for spec in rmg.initialSpecies:
        spec.thermo = processThermoData(spec, spec.thermo)
        if not spec.reactive:
            rmg.reactionModel.enlarge(spec)
    for spec in rmg.initialSpecies:
        if spec.reactive:
            rmg.reactionModel.enlarge(spec)
    logging.info("isotope: Finalizing the species additions")
    rmg.initializeReactionThresholdAndReactFlags()
    rmg.reactionModel.initializeIndexSpeciesDict()


def generateIsotopeModel(outputDirectory, rmg0, isotopes):
    """
    Replace the core species of the rmg model with the parameter list
    of species.

    Generate all reactions between new list of core species.

    Returns created RMG object.
    """
    logging.info("isotope: creating the RMG model")
    rmg = RMG(inputFile=rmg0.inputFile, outputDirectory=outputDirectory)
    rmg.attach(ChemkinWriter(outputDirectory))

    logging.info("isotope: making the isotope model for with all species")
    initializeIsotopeModel(rmg, isotopes)

    logging.info("isotope: enlarging the isotope model")
    rmg.reactionModel.enlarge(reactEdge=True,
        unimolecularReact=rmg.unimolecularReact,
        bimolecularReact=rmg.bimolecularReact)

    # alter forward reaction rates to be thermodynamically consistant
    # across isotopomers
    correctAFactors(rmg.reactionModel.core.reactions)

    logging.info("isotope: saving files")
    rmg.saveEverything()

    rmg.finish()     

    return rmg   

def generateIsotopomers(spc, N=1):
    """
    Generate all isotopomers of the parameter species by adding max. N carbon isotopes to the
    atoms of the species.
    """

    mol = spc.molecule[0]
    isotope = getElement(6, 13)
    carbons = filter(lambda at: at.symbol == isotope.symbol, mol.atoms)
    
    mols = []
    addIsotope(0, N, mol, mols, isotope)

    spcs = []
    for isomol in mols:
        isotopomer = Species(molecule=[isomol], thermo=spc.thermo, transportData=spc.transportData, reactive=spc.reactive)
        isotopomer.generateResonanceIsomers(keepIsomorphic=True)
        spcs.append(isotopomer)

    # do not retain identical species:
    filtered = []
    while spcs:
        candidate = spcs.pop()
        unique = True
        for isotopomer in filtered:
            if isotopomer.isIsomorphic(candidate):
                unique = False
                break
        if unique: filtered.append(candidate)

    for isotopomer in filtered:
        correctEntropy(isotopomer, spc)

    return filtered

def addIsotope(i, N, mol, mols, element):
    """
    
    Iterate over the atoms of the molecule, and changes the element object
    of the atom by the provide parameter element object. Add the newly created
    isotopomer to the list of Molecule objects. For each created isotopomer,
    recursively call the method, until the maximum number of isotopes per molecule
    (N) is reached.

    """
    if i == N: return
    else:
        atoms = filter(lambda at: at.symbol == element.symbol, mol.atoms)
        for at in atoms:
            if at.element == element: continue
            else:
                isotopomer = mol.copy(deep=True)
                isotopomer.atoms[mol.atoms.index(at)].element = element
                mols.append(isotopomer)
                addIsotope(i+1, N, isotopomer, mols, element)


def solve(rmg):
    """
    Solve the reaction system, read the simulation 
    profiles, and return them into a pandas dataframe.
    """

    solverDir = os.path.join(rmg.outputDirectory, 'solver')
    try:
        shutil.rmtree(solverDir)
    except OSError, e:
        pass
    
    os.mkdir(solverDir)

    reactionSysIndex = 0
    listener = SimulationProfileWriter(rmg.outputDirectory, reactionSysIndex, rmg.reactionModel.core.species)
    
    reactionSystem = rmg.reactionSystems[0]
    reactionSystem.attach(listener)

    reactionModel = rmg.reactionModel

    # run simulation:
    terminated, obj = reactionSystem.simulate(
        coreSpecies = reactionModel.core.species,
        coreReactions = reactionModel.core.reactions,
        edgeSpecies = reactionModel.edge.species,
        edgeReactions = reactionModel.edge.reactions,
        toleranceKeepInEdge = 0,
        toleranceMoveToCore = 1,
        toleranceInterruptSimulation = 1,
    ) 

    simCSV = os.path.join(rmg.outputDirectory, 'solver/simulation_{}_{}.csv'.format(reactionSysIndex + 1, len(reactionModel.core.species)))
    spcdata = pd.read_csv(simCSV)
    
    return spcdata

def cluster(objList):
    """
    Creates subcollections of isotopomers/reactions that 
    only differ in their isotopic labeling.

    This method works for either species or reactions.

    It is O(n^2) efficient
    """

    unclustered = copy(objList)

    # [[list of Species objs]]
    clusters = []

    while unclustered:
        candidate = unclustered.pop()
        for cluster in clusters:
            if compareIsotopomers(cluster[0],candidate):
                cluster.append(candidate)
                break
        else:
            clusters.append([candidate])

    return clusters

def removeIsotope(labeledObj, inplace = False):
    """
    Create a deep copy of the first molecule of the species object and replace
    non-normal Element objects (of special isotopes) by the 
    expected isotope.

    If the boolean `inplace` is True, the method remove the isotopic atoms of 
    the Species/Reaction
    inplace and returns a list of atom objects & element pairs for adding back
    to the oritinal object. This should significantly improve speed of this method.

    If successful, the non-inplace parts should be removed
    """
    if isinstance(labeledObj,Species):
        if inplace:
            modifiedAtoms = []
            for mol in labeledObj.molecule:
                for atom in mol.atoms:
                    if atom.element.isotope != -1:
                        modifiedAtoms.append((atom,atom.element))
                        atom.element = getElement(atom.element.symbol)
            return modifiedAtoms
        else:
            stripped = labeledObj.copy(deep=True)
    
            for atom in stripped.molecule[0].atoms:
                if atom.element.isotope != -1:
                    atom.element = getElement(atom.element.symbol)
    
        # only do it for the first molecule, generate the other resonance isomers.
            stripped.molecule = [stripped.molecule[0]]
            stripped.generateResonanceIsomers()
    
        return stripped

    elif isinstance(labeledObj,Reaction):
        
        if inplace:
            
            atomList = []
            for reactant in  labeledObj.reactants:
                removed = removeIsotope(reactant,inplace)
                if removed:
                    atomList += removed
            for product in labeledObj.products:
                removed = removeIsotope(product,inplace)
                if removed:
                    atomList += removed

            return atomList
        else:
            strippedRxn = labeledObj.copy()

            strippedReactants = []
            for reactant in  strippedRxn.reactants:
                strippedReactants.append(removeIsotope(reactant,inplace))
            strippedRxn.reactants = strippedReactants

            strippedProducts = []
            for product in  strippedRxn.products:
                strippedProducts.append(removeIsotope(product,inplace))
            strippedRxn.products = strippedProducts

            return strippedRxn
    else:
        raise TypeError('Only Reaction and Species objects are supported')

def redoIsotope(atomList):
    """
    This takes a list of zipped atoms with their isotopes removed, from 
    and elements.
    """
    for atom, element in atomList:
        atom.element = element

def compareIsotopomers(obj1, obj2):
    """
    This method takes two species or reaction objects and returns true if
    they only differ in isotopic labeling, and false if they have other
    differences.

    The removeIsotope method can be slow, especially when comparing molecules
    and reactions. This was due to many copying of objects.

    This method avoid copying by storing the isotope and atom objects,
    removing them, doing the comparison, and rewriting them when
    finished the comparison.
    """

    atomlist = removeIsotope(obj1,inplace=True) + removeIsotope(obj2,inplace=True)
    comparisonBool = obj1.isIsomorphic(obj2)
    redoIsotope(atomlist)
    return comparisonBool

def retrieveConcentrations(spcdata, clusters):
    """
    Iterate over the species in the list of clustered species
    and return a dataframe, but filled with 
    concentration columns of the corresponding species.
    """

    concs = []

    for cluster in clusters:
        df = pd.DataFrame()
        for spc in cluster:
            try:
                header = '{}({})'.format(spc.label, spc.index)
                df[header] = spcdata[header]
            except KeyError, e:
                header = '{}'.format(spc.label)
                try:
                    df[header] = spcdata[header]
                except KeyError, e:
                    raise e
            
        concs.append(df)

    return concs

def computeProbabilities(df):
    """
    Compute the isotope probabilities by dividing the 
    species concentration by the sum of the clustered species concentrations.
    """
    probs = []
    sumConcs = df.sum(axis=1)
    for column in df:
        df[column] = df[column] / sumConcs

    return df

def generateRMGModel(inputFile, outputDirectory):
    """
    Generate the RMG-Py model NOT containing any non-normal isotopomers.

    Returns created RMG object.
    """
    initializeLog(logging.INFO, os.path.join(outputDirectory, 'RMG.log'))
    # generate mechanism:
    rmg = RMG(inputFile = os.path.abspath(inputFile),
            outputDirectory = os.path.abspath(outputDirectory)
        )
    rmg.execute()

    return rmg

def correctEntropy(isotopomer, isotopeless):
    """
    Correct the entropy of the isotopomer by the following correction for symmetry:

    S(corrected) = S(original) + R*ln(sigma(isotopeless)) - R*ln(sigma(isotopomer))
        
    This method also copies the Enthalpy, Cp and other thermo parameters from isotopeless
    """

    # calculate -R ln (sigma) in SI units (J/K/mol)
    Sisotopeless = - constants.R * math.log(isotopeless.getSymmetryNumber())
    Sisotopomer = - constants.R * math.log(isotopomer.getSymmetryNumber())

    # convert species thermo to ThermoData object:
    nasa = isotopomer.thermo

    # apply correction to entropy at 298K
    deltaS = Sisotopomer - Sisotopeless
    nasa = nasa.changeBaseEntropy(deltaS)

    # put the corrected thermo back as a species attribute:
    isotopomer.thermo = nasa

def correctAFactors(reactions):
    """
    This method corrects the A factors of reactions
    to be consistant with the thermodynamics of reactants
    which leads to a more stable isotopomer ratio.It takes
    in a list of all core reactions (after model generation)
    and modifies the A factors of the reactions. The reactions
    are modified in place (no returning various isotopomers)
    """
    reactionClusters = cluster(reactions)
    for rxnList in reactionClusters:
        correctAFactorsOfIsotopomers(rxnList)

    # halfing Afactors when reactants are identical 
    from rmgpy.kinetics.arrhenius import Arrhenius, ArrheniusEP
    for rxn in rxnList:
        if len(rxn.reactants) == 2:
            if rxn.reactants[0].isIsomorphic(rxn.reactants[1]):
                if isinstance(rxn.kinetics,Arrhenius) or isinstance(rxn.kinetics,ArrheniusEP):
                    rxn.kinetics.A.value = rxn.kinetics.A.value / 2

def correctAFactorsOfIsotopomers(rxnList):
    """
    Since different isotopomers sometimes have different symmetry
    numbers (and the TS of both often don't), the rates of the isotopomer
    reactions may not be correct. This method seeks to correct for this
    discrpency by taking in a list of identical reactions varying in labeling 
    (which can be obtained from `cluster`), and modifying the A-factors 
    to be thermodynamically equivalent.
    
    `rxnList` - a list of identical reactions varying in labeling
    
    Symmetry difference = symmetry labeled rxn / symmetry unlabeled rxn
    A(labeled) = A(non-labeled) * symmetry difference
    """
    unlabeledRxn = None
    for rxn in rxnList:
        if not isEnriched(rxn):
            unlabeledRxn = rxn
            break
    if unlabeledRxn is None:
        logging.info('No unlabeled reaction sent to correctAFactorsForIsotopomers. The reactions are of type {}'.format(str(rxnList[0])))
        unlabeledRxn = removeIsotope(rxnList[0])
        # note the kinetics will be off when comparing these, but it will
        # be thermodynamically consistant
    
    unlabeledSymmetry = __getReactionSymmetryNumber(unlabeledRxn)
    unlabeledA = unlabeledRxn.kinetics.A.value_si
    for rxn in rxnList:
        symmetry = __getReactionSymmetryNumber(rxn)
        AFactor = unlabeledRxn.kinetics.A.value_si
        
        symmetryRatio = symmetry / unlabeledSymmetry
        AFactorRatio = AFactor / unlabeledA
        
        if not np.isclose(symmetryRatio,AFactorRatio):
            logging.info("reaction {} initially had incorrect AFactor. Making it match unlabeled reaction {}".format(str(rxn.index),str(unlabeledRxn.index)))
            AFactorMultiplier = symmetry / unlabeledSymmetry
            rxn.kinetics.A.value = unlabeledRxn.kinetics.A.value * AFactorMultiplier
    
def isEnriched(obj):
    """
    Returns True if the species or reaction object has any enriched isotopes.
    """
    
    if isinstance(obj,Species):
        for atom in obj.molecule[0].atoms:
            if atom.element.isotope != -1 and not np.allclose(atom.element.mass, getElement(atom.element.symbol).mass):
                return True
        return False
    elif isinstance(obj,Reaction):
        enriched = []
        for spec in obj.reactants:
            enriched.append(isEnriched(spec))
        for spec in obj.products:
            enriched.append(isEnriched(spec))
        return any(enriched)
    else:
        raise TypeError('isEnriched only takes species and reaction objects. {} was sent'.format(str(type(obj))))
def __getReactionSymmetryNumber(reaction):
    """
    This method finds the reaction symmetry number for a reaction. The procedure 
    is slightly modified to account for no transition state, by replacing it with
    product symmetry.
    
    When identical reactants are present, the algorythm only accounts for them
    if they come from reaction family `R_Recombination`. Other reaction
    families already account for that rate difference in degeneracy caluclations
    
    rxn symmetry number = reactant symmetry / product symmetry
    """
    reactantSym = 1.
    for reactant in reaction.reactants:
        reactantSym *= reactant.getSymmetryNumber()
    
    productSym = 1.
    for product in reaction.products:
        productSym *= product.getSymmetryNumber()
        
    return reactantSym / productSym
        
    
def run(inputFile, isotopeInputFile, outputDir, original=None, isotopeLoc=None):
    """
    Accepts two input files, one input file with the RMG-Py model to generate, NOT
    containing any non-normal isotopomers, and one input file for the model to be 
    generated starting from the isotopomers generated from the core species of the first model.

    Firstly, generates the RMG model for the first input file. Takes the core species of that mechanism
    and generates all isotopomers of those core species. Next, generates all reactions between the
    generated pool of isotopomers, and writes it to file. Next, reads in that newly generated mechanism
    and runs a simulation with the given conditions of the second input file, writes species mole fractions
    to file. Next, clusters the isotopomers together again, so that isotopomer probabilities can be computed.

    Returns:
    - a pandas data frame with isotopomer probabilities as a function of reaction time 
    - a list of species objects corresponding to the isotopomers of the generated model
    - a pandas data frame with isotopomer speciation data as a function of reaction time
    """
    logging.info("isotope: Starting the RMG isotope generation method 'run'")
    if not original:
        logging.info("isotope: original model not found, generating new one in directory `rmg`")
        outputdirRMG = os.path.join(outputDir, 'rmg')
        os.mkdir(outputdirRMG)

        rmg = generateRMGModel(inputFile, outputdirRMG)
    else:
        logging.info("isotope: original model being copied from previous RMG job in folder {}".format(original))
        outputdirRMG = original
        chemkinFile = os.path.join(outputdirRMG, 'chemkin', 'chem_annotated.inp')
        dictFile = os.path.join(outputdirRMG, 'chemkin', 'species_dictionary.txt')
        rmg = loadRMGJob(inputFile, chemkinFile, dictFile, generateImages=False, useChemkinNames=True)

    isotopeInputFile = os.path.abspath(isotopeInputFile)
    
    logging.info("isotope: creating RMG isotope input file")
    rmgIso = RMG(inputFile = isotopeInputFile)
    logging.info("isotope: loading isotope input file")
    rmgIso.loadInput(rmgIso.inputFile)

    if not isotopeLoc:
        logging.info("isotope: isotope model not found, generating new model")
        print('Generating isotopomers for the core species in {}'.format(outputdirRMG))
        isotopes = []

        try:
            speciesConstraints = getInput('speciesConstraints')
        except Exception, e:
            logging.debug('Species constraints could not be found.')
            raise e

        try:
            maxIsotopes = speciesConstraints['maximumIsotopicAtoms']
        except KeyError, e:
            print ('Could not find the maxIsotopicAtoms constraint in the input file. Exiting...')
            raise e
        logging.info("isotope: adding all the new isotopomers")
        for spc in rmg.reactionModel.core.species:
            findCp0andCpInf(spc, spc.thermo)
            isotopes.extend(generateIsotopomers(spc, maxIsotopes))

        logging.info("isotope: adding original species to the model")
        # add the original unlabeled species:
        isotopes.extend(rmg.reactionModel.core.species)
        logging.info('Number of isotopomers: {}'.format(len(isotopes)))

        outputdirIso = os.path.join(outputDir, 'iso')
        os.mkdir(outputdirIso)

        logging.info('isotope: Generating RMG isotope model in {}'.format(outputdirIso))
        rmgIso = generateIsotopeModel(outputdirIso, rmg, isotopes)
    else:
        outputdirIso= isotopeLoc

    chemkinFileIso = os.path.join(outputdirIso, 'chemkin', 'chem_annotated.inp')
    dictFileIso = os.path.join(outputdirIso, 'chemkin', 'species_dictionary.txt')

    logging.info('isotope: Loading isotope chemkin model.\nInput file: {}\nChemkin file: {}\nDict file: {}'\
        .format(isotopeInputFile, chemkinFileIso, dictFileIso))

    rmgIso = loadRMGJob(isotopeInputFile, chemkinFileIso, dictFileIso, generateImages=True, useChemkinNames=True)
    rmgIso.outputDirectory = outputdirIso

    logging.info('isotope: Clustering isotopomers...')
    clusters = cluster(rmgIso.reactionModel.core.species)

    logging.info('isotope: Solving the Isotope model with the isotope input file...')
    spcData = solve(rmgIso)    
    
    logging.info('isotope: Generating concentrations for lumped species...')
    concs = retrieveConcentrations(spcData, clusters)

    logging.info('isotope: Computing isotopomer probabilities...')
    probs = []
    for df in concs:
        df = computeProbabilities(df.copy())
        probs.append(df)

    return probs, rmgIso.reactionModel.core.species, spcData
