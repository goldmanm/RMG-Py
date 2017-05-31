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
from copy import copy, deepcopy
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
from rmgpy.data.rmg import getDB

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
    logging.debug("isotope: called generateIsotopeModel")
    rmg = RMG(inputFile=rmg0.inputFile, outputDirectory=outputDirectory)
    rmg.attach(ChemkinWriter(outputDirectory))

    logging.info("isotope: making the isotope model for with all species")
    initializeIsotopeModel(rmg, isotopes)

    logging.info("isotope: enlarging the isotope model")
    rmg.reactionModel.enlarge(reactEdge=True,
        unimolecularReact=rmg.unimolecularReact,
        bimolecularReact=rmg.bimolecularReact)

    logging.info("isotope: clustering reactions")
    clusters = cluster(rmg.reactionModel.core.reactions)
    logging.info('isotope: fixing the directions of every reaction to a standard')
    for isotopomerRxnList in clusters:
        ensureReactionDirection(isotopomerRxnList)
        
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
        isotopomer = Species(molecule=[isomol], thermo=deepcopy(spc.thermo), transportData=spc.transportData, reactive=spc.reactive)
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
    elif isinstance(labeledObj,Molecule):
        if inplace:
            modifiedAtoms = []
            for atom in labeledObj.atoms:
                if atom.element.isotope != -1:
                    modifiedAtoms.append((atom,atom.element))
                    atom.element = getElement(atom.element.symbol)
            return modifiedAtoms
        else:
            stripped = labeledObj.copy(deep=True)
    
            for atom in stripped.atoms:
                if atom.element.isotope != -1:
                    atom.element = getElement(atom.element.symbol)

            return stripped
    else:
        raise TypeError('Only Reaction, Species, and Molecule objects are supported')

def ensureReactionDirection(isotopomerRxns):
    """
    given a list of reactions with varying isotope labels but identical structure,
    obtained from the `cluster` method, this method remakes the kinetics so that 
    they all face the same direction.
    """
    
    # find isotopeless reaction as standard
    reference = isotopomerRxns[0]
    family = getDB('kinetics').families[reference.family]
    if family.ownReverse:
        for rxn in isotopomerRxns:
            if not compareIsotopomers(rxn, reference, eitherDirection=False):
                # the reaction is in the oposite direction
                logging.info('isotope: identified flipped reaction direction in reaction number {} of reaction {}. Altering the direction.'.format(rxn.index, str(rxn)))
                # reverse reactants and products
                rxn.reactants, rxn.products = rxn.products, rxn.reactants
                rxn.pairs = [(p,r) for r,p in rxn.pairs]

                # calculateDegeneracy
                rxnMols = TemplateReaction(reactants = [spec.molecule[0] for spec in rxn.reactants],
                                           products = [spec.molecule[0] for spec in rxn.products])
                forwardDegen = family.calculateDegeneracy(rxnMols)

                # set degeneracy to isotopeless reaction
                rxn.degeneracy = reference.degeneracy
                # make this reaction have kinetics of isotopeless reaction
                newKinetics = deepcopy(reference.kinetics)
                rxn.kinetics = newKinetics

                # set degeneracy to new reaction
                rxn.degeneracy = forwardDegen


def redoIsotope(atomList):
    """
    This takes a list of zipped atoms with their isotopes removed, from 
    and elements.
    """
    for atom, element in atomList:
        atom.element = element

def compareIsotopomers(obj1, obj2, eitherDirection = True):
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
    if isinstance(obj1,Reaction):
        comparisonBool = obj1.isIsomorphic(obj2, eitherDirection)
    elif isinstance(obj1,Species):
        comparisonBool = obj1.isIsomorphic(obj2)
    else:
        raise TypeError('Only Reaction and Speicies Objects are supported in compareIsotopomers')
    redoIsotope(atomlist)
    return comparisonBool

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
        
    
    
def __compareReactionSymmetry(labeled_rxn, unlabeld_rxn):
    """
    compares the symmetry number of two reactions which only vary in labeling
    of isotopes.. Since there is no transition
    state, this method assumes that the transition state's symmetry is only
    decreased when a reactant and it's flux pair both have lower symmetry number.
    If a change in symmetry occurs only in reactant or product, the TS symmetry
    is assumed not to change (since that symmery would be broken in the TS as well)
    """
    A_factor_modification = 1
    for pair in labeled_rxn.pairs:
        for pair2 in labeled_rxn.pairs:
            # if the pair is corresponding
            if all([compareIsotopomers(pair[0],pair2[0]), compareIsotopomers(pair[1],pair2[1])]):
                reactant_symmetry_ratio = float(pair[0].getSymmetryNumber()) / pair2[0].getSymmetryNumber()
                # if the reactant symmetry changes
                if abs(reactant_symmetry_ratio - 1) > 1e-5:
                    product_symmetry_ratio = float(pair[1].getSymmetryNumber()) / pair2[1].getSymmetryNumber()
                    if product_symmetry_ratio != reactant_symmetry_ratio:
                        A_factor_modification *= reactant_symmetry_ratio
                    
def run(inputFile, outputDir, original=None):
    """
    Accepts one input file with the RMG-Py model to generate. This file can contain 
    the optional parameter 'maximumIsotopicAtoms' to reduce isotope labeling.

    Firstly, generates the RMG model for the first input file. Takes the core species of that mechanism
    and generates all isotopomers of those core species. Next, generates all reactions between the
    generated pool of isotopomers, and writes it to file. 
    """
    logging.info("isotope: Starting the RMG isotope generation method 'run'")
    if not original:
        logging.info("isotope: original model not found, generating new one in directory `rmg`")
        logging.info("isotope: check `rmg/RMG.log` for the rest of the logging info.")

        outputdirRMG = os.path.join(outputDir, 'rmg')
        os.mkdir(outputdirRMG)

        rmg = generateRMGModel(inputFile, outputdirRMG)
    else:
        logging.info("isotope: original model being copied from previous RMG job in folder {}".format(original))
        outputdirRMG = original
        chemkinFile = os.path.join(outputdirRMG, 'chemkin', 'chem_annotated.inp')
        dictFile = os.path.join(outputdirRMG, 'chemkin', 'species_dictionary.txt')
        rmg = loadRMGJob(inputFile, chemkinFile, dictFile, generateImages=False, useChemkinNames=True)

    logging.info("isotope: generating isotope model")
    logging.info('Generating isotopomers for the core species in {}'.format(outputdirRMG))
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
    generateIsotopeModel(outputdirIso, rmg, isotopes)
