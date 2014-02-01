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

"""

import os.path
import math
import logging
import numpy
from copy import copy, deepcopy

from base import Database, Entry, makeLogicNode, DatabaseError

import rmgpy.constants as constants
from rmgpy.thermo import NASAPolynomial, NASA, ThermoData, Wilhoit
from rmgpy.molecule import Molecule, Atom, Bond, Group
import rmgpy.molecule

################################################################################

def saveEntry(f, entry):
    """
    Write a Pythonic string representation of the given `entry` in the thermo
    database to the file object `f`.
    """
    
    f.write('entry(\n')
    f.write('    index = {0:d},\n'.format(entry.index))
    f.write('    label = "{0}",\n'.format(entry.label))

    if isinstance(entry.item, Molecule):
        f.write('    molecule = \n')
        f.write('"""\n')
        f.write(entry.item.toAdjacencyList(removeH=False))
        f.write('""",\n')
    elif isinstance(entry.item, Group):
        f.write('    group = \n')
        f.write('"""\n')
        f.write(entry.item.toAdjacencyList())
        f.write('""",\n')
    else:
        f.write('    group = "{0}",\n'.format(entry.item))

    if isinstance(entry.data, ThermoData):
        f.write('    thermo = ThermoData(\n')
        f.write('        Tdata = {0!r},\n'.format(entry.data.Tdata))
        f.write('        Cpdata = {0!r},\n'.format(entry.data.Cpdata))
        f.write('        H298 = {0!r},\n'.format(entry.data.H298))
        f.write('        S298 = {0!r},\n'.format(entry.data.S298))
        if entry.data.Tmin is not None: f.write('        Tmin = {0!r},\n'.format(entry.data.Tmin))
        if entry.data.Tmax is not None: f.write('        Tmax = {0!r},\n'.format(entry.data.Tmax))
        f.write('    ),\n')
    elif isinstance(entry.data, Wilhoit):
        f.write('    thermo = Wilhoit(\n')
        f.write('        cp0 = {0!r},\n'.format(entry.data.cp0))
        f.write('        cpInf = {0!r},\n'.format(entry.data.cpInf))
        f.write('        a0 = {0:g},\n'.format(entry.data.a0))
        f.write('        a1 = {0:g},\n'.format(entry.data.a1))
        f.write('        a2 = {0:g},\n'.format(entry.data.a2))
        f.write('        a3 = {0:g},\n'.format(entry.data.a3))
        f.write('        B = {0!r},\n'.format(entry.data.B))
        f.write('        H0 = {0!r},\n'.format(entry.data.H0))
        f.write('        S0 = {0!r},\n'.format(entry.data.S0))
        if entry.data.Tmin is not None: f.write('        Tmin = {0!r},\n'.format(entry.data.Tmin))
        if entry.data.Tmax is not None: f.write('        Tmax = {0!r},\n'.format(entry.data.Tmax))
        f.write('    ),\n')
    elif isinstance(entry.data, NASA):
        f.write('    thermo = NASA(\n')
        f.write('        polynomials = [\n')
        for poly in entry.data.polynomials:
            f.write('            {0!r},\n'.format(poly))
        f.write('        ],\n')
        if entry.data.Tmin is not None: f.write('        Tmin = {0!r},\n'.format(entry.data.Tmin))
        if entry.data.Tmax is not None: f.write('        Tmax = {0!r},\n'.format(entry.data.Tmax))
        f.write('    ),\n')
    else:
        f.write('    thermo = {0!r},\n'.format(entry.data))

    if entry.reference is not None: f.write('    reference = {0!r},\n'.format(entry.reference))
    if entry.referenceType != "": f.write('    referenceType = "{0}",\n'.format(entry.referenceType))
    f.write('    shortDesc = u"""')
    f.write(entry.shortDesc)
    f.write('""",\n')
    f.write('    longDesc = \n')
    f.write('u"""\n')
    f.write(entry.longDesc.strip() + "\n")
    f.write('""",\n')

    f.write('    history = [\n')
    for time, user, action, description in entry.history:
        f.write('        ("{0}","{1}","{2}","""{3}"""),\n'.format(time, user, action, description))
    f.write('    ],\n')

    f.write(')\n\n')

def generateOldLibraryEntry(data):
    """
    Return a list of values used to save entries to the old-style RMG
    thermo database based on the thermodynamics object `data`.
    """
    if isinstance(data, ThermoData):
        return '{0:9g} {1:9g} {2:9g} {3:9g} {4:9g} {5:9g} {6:9g} {7:9g} {8:9g} {9:9g} {10:9g} {11:9g}'.format(
            data.H298.value_si/4184.,
            data.S298.value_si/4.184,
            data.Cpdata.value_si[0]/4.184,
            data.Cpdata.value_si[1]/4.184,
            data.Cpdata.value_si[2]/4.184,
            data.Cpdata.value_si[3]/4.184,
            data.Cpdata.value_si[4]/4.184,
            data.Cpdata.value_si[5]/4.184,
            data.Cpdata.value_si[6]/4.184,
            data.H298.uncertainty/4184.,
            data.S298.uncertainty/4.184,
            max(data.Cpdata.uncertainty)/4.184,
        )
    elif isinstance(data, basestring):
        return data
    else:
        return '{0:9g} {1:9g} {2:9g} {3:9g} {4:9g} {5:9g} {6:9g} {7:9g} {8:9g} {9:9g} {10:9g} {11:9g}'.format(
            data.getEnthalpy(298)/4184.,
            data.getEntropy(298)/4.184,
            data.getHeatCapacity(300)/4.184,
            data.getHeatCapacity(400)/4.184,
            data.getHeatCapacity(500)/4.184,
            data.getHeatCapacity(600)/4.184,
            data.getHeatCapacity(800)/4.184,
            data.getHeatCapacity(1000)/4.184,
            data.getHeatCapacity(1500)/4.184,
            0,
            0,
            0,
        )

def processOldLibraryEntry(data):
    """
    Process a list of parameters `data` as read from an old-style RMG
    thermo database, returning the corresponding thermodynamics object.
    """
    return ThermoData(
        Tdata = ([300,400,500,600,800,1000,1500],"K"),
        Cpdata = ([float(d) for d in data[2:9]],"cal/(mol*K)","+|-",float(data[11])),
        H298 = (float(data[0]),"kcal/mol","+|-",float(data[9])),
        S298 = (float(data[1]),"cal/(mol*K)","+|-",float(data[10])),
    )


################################################################################

class ThermoDepository(Database):
    """
    A class for working with the RMG thermodynamics depository.
    """

    def __init__(self, label='', name='', shortDesc='', longDesc=''):
        Database.__init__(self, label=label, name=name, shortDesc=shortDesc, longDesc=longDesc)

    def loadEntry(self, index, label, molecule, thermo, reference=None, referenceType='', shortDesc='', longDesc='', history=None):
        entry = Entry(
            index = index,
            label = label,
            item = Molecule().fromAdjacencyList(molecule),
            data = thermo,
            reference = reference,
            referenceType = referenceType,
            shortDesc = shortDesc,
            longDesc = longDesc.strip(),
            history = history or [],
        )
        self.entries[label] = entry
        return entry

    def saveEntry(self, f, entry):
        """
        Write the given `entry` in the thermo database to the file object `f`.
        """
        return saveEntry(f, entry)

################################################################################

class ThermoLibrary(Database):
    """
    A class for working with a RMG thermodynamics library.
    """

    def __init__(self, label='', name='', shortDesc='', longDesc=''):
        Database.__init__(self, label=label, name=name, shortDesc=shortDesc, longDesc=longDesc)

    def loadEntry(self,
                  index,
                  label,
                  molecule,
                  thermo,
                  reference=None,
                  referenceType='',
                  shortDesc='',
                  longDesc='',
                  history=None
                  ):
        self.entries[label] = Entry(
            index = index,
            label = label,
            item = Molecule().fromAdjacencyList(molecule),
            data = thermo,
            reference = reference,
            referenceType = referenceType,
            shortDesc = shortDesc,
            longDesc = longDesc.strip(),
            history = history or [],
        )

    def saveEntry(self, f, entry):
        """
        Write the given `entry` in the thermo database to the file object `f`.
        """
        return saveEntry(f, entry)

    def generateOldLibraryEntry(self, data):
        """
        Return a list of values used to save entries to the old-style RMG
        thermo database based on the thermodynamics object `data`.
        """
        return generateOldLibraryEntry(data)

    def processOldLibraryEntry(self, data):
        """
        Process a list of parameters `data` as read from an old-style RMG
        thermo database, returning the corresponding thermodynamics object.
        """
        return processOldLibraryEntry(data)

################################################################################

class ThermoGroups(Database):
    """
    A class for working with an RMG thermodynamics group additivity database.
    """

    def __init__(self, label='', name='', shortDesc='', longDesc=''):
        Database.__init__(self, label=label, name=name, shortDesc=shortDesc, longDesc=longDesc)

    def loadEntry(self,
                  index,
                  label,
                  group,
                  thermo,
                  reference=None,
                  referenceType='',
                  shortDesc='',
                  longDesc='',
                  history=None
                  ):
        if group[0:3].upper() == 'OR{' or group[0:4].upper() == 'AND{' or group[0:7].upper() == 'NOT OR{' or group[0:8].upper() == 'NOT AND{':
            item = makeLogicNode(group)
        else:
            item = Group().fromAdjacencyList(group)
        self.entries[label] = Entry(
            index = index,
            label = label,
            item = item,
            data = thermo,
            reference = reference,
            referenceType = referenceType,
            shortDesc = shortDesc,
            longDesc = longDesc.strip(),
            history = history or [],
        )
    
    def saveEntry(self, f, entry):
        """
        Write the given `entry` in the thermo database to the file object `f`.
        """
        return saveEntry(f, entry)

    def generateOldLibraryEntry(self, data):
        """
        Return a list of values used to save entries to the old-style RMG
        thermo database based on the thermodynamics object `data`.
        """
        
        return generateOldLibraryEntry(data)

    def processOldLibraryEntry(self, data):
        """
        Process a list of parameters `data` as read from an old-style RMG
        thermo database, returning the corresponding thermodynamics object.
        """
        return processOldLibraryEntry(data)

################################################################################

class ThermoDatabase(object):
    """
    A class for working with the RMG thermodynamics database.
    """

    def __init__(self):
        self.depository = {}
        self.libraries = {}
        self.groups = {}
        self.libraryOrder = []
        self.local_context = {
            'ThermoData': ThermoData,
            'Wilhoit': Wilhoit,
            'NASAPolynomial': NASAPolynomial,
            'NASA': NASA,
        }
        self.global_context = {}

    def __reduce__(self):
        """
        A helper function used when pickling a ThermoDatabase object.
        """
        d = {
            'depository': self.depository,
            'libraries': self.libraries,
            'groups': self.groups,
            'libraryOrder': self.libraryOrder,
        }
        return (ThermoDatabase, (), d)

    def __setstate__(self, d):
        """
        A helper function used when unpickling a ThermoDatabase object.
        """
        self.depository = d['depository']
        self.libraries = d['libraries']
        self.groups = d['groups']
        self.libraryOrder = d['libraryOrder']

    def load(self, path, libraries=None, depository=True):
        """
        Load the thermo database from the given `path` on disk, where `path`
        points to the top-level folder of the thermo database.
        """
        if depository:
            self.loadDepository(os.path.join(path, 'depository'))
        else:
            self.depository = {}
        self.loadLibraries(os.path.join(path, 'libraries'), libraries)
        self.loadGroups(os.path.join(path, 'groups'))
        
    def loadDepository(self, path):
        """
        Load the thermo database from the given `path` on disk, where `path`
        points to the top-level folder of the thermo database.
        """
        self.depository = {}
        self.depository['stable']  = ThermoDepository().load(os.path.join(path, 'stable.py'), self.local_context, self.global_context)
        self.depository['radical'] = ThermoDepository().load(os.path.join(path, 'radical.py'), self.local_context, self.global_context)
        self.depository['HNO'] = ThermoDepository().load(os.path.join(path, 'HNO.py'), self.local_context, self.global_context)

    def loadLibraries(self, path, libraries=None):
        """
        Load the thermo database from the given `path` on disk, where `path`
        points to the top-level folder of the thermo database.
        """
        self.libraries = {}; self.libraryOrder = []
        for (root, dirs, files) in os.walk(os.path.join(path)):
            for f in files:
                name, ext = os.path.splitext(f)
                if ext.lower() == '.py' and (libraries is None or name in libraries):
                    logging.info('Loading thermodynamics library from {0} in {1}...'.format(f, root))
                    library = ThermoLibrary()
                    library.load(os.path.join(root, f), self.local_context, self.global_context)
                    library.label = os.path.splitext(f)[0]
                    self.libraries[library.label] = library
                    self.libraryOrder.append(library.label)
        if libraries is not None:
            self.libraryOrder = libraries

    def loadGroups(self, path):
        """
        Load the thermo database from the given `path` on disk, where `path`
        points to the top-level folder of the thermo database.
        """
        logging.info('Loading thermodynamics group database from {0}...'.format(path))
        self.groups = {}
        self.groups['group']   =   ThermoGroups(label='group').load(os.path.join(path, 'group.py'  ), self.local_context, self.global_context)
        self.groups['gauche']  =  ThermoGroups(label='gauche').load(os.path.join(path, 'gauche.py' ), self.local_context, self.global_context)
        self.groups['int15']   =   ThermoGroups(label='int15').load(os.path.join(path, 'int15.py'  ), self.local_context, self.global_context)
        self.groups['ring']    =    ThermoGroups(label='ring').load(os.path.join(path, 'ring.py'   ), self.local_context, self.global_context)
        self.groups['radical'] = ThermoGroups(label='radical').load(os.path.join(path, 'radical.py'), self.local_context, self.global_context)
        self.groups['polycyclic'] = ThermoGroups(label='polycyclic').load(os.path.join(path, 'polycyclic.py'), self.local_context, self.global_context)
        self.groups['other']   =   ThermoGroups(label='other').load(os.path.join(path, 'other.py'  ), self.local_context, self.global_context)
        self.groups['HNO']   =   ThermoGroups(label='HNO').load(os.path.join(path, 'HNO.py'  ), self.local_context, self.global_context)

    def save(self, path):
        """
        Save the thermo database to the given `path` on disk, where `path`
        points to the top-level folder of the thermo database.
        """
        path = os.path.abspath(path)
        if not os.path.exists(path): os.mkdir(path)
        self.saveDepository(os.path.join(path, 'depository'))
        self.saveLibraries(os.path.join(path, 'libraries'))
        self.saveGroups(os.path.join(path, 'groups'))

    def saveDepository(self, path):
        """
        Save the thermo depository to the given `path` on disk, where `path`
        points to the top-level folder of the thermo depository.
        """
        if not os.path.exists(path): os.mkdir(path)
        self.depository['stable'].save(os.path.join(path, 'stable.py'))
        self.depository['radical'].save(os.path.join(path, 'radical.py'))

    def saveLibraries(self, path):
        """
        Save the thermo libraries to the given `path` on disk, where `path`
        points to the top-level folder of the thermo libraries.
        """
        if not os.path.exists(path): os.mkdir(path)
        for library in self.libraries.values():
            library.save(os.path.join(path, '{0}.py'.format(library.label)))

    def saveGroups(self, path):
        """
        Save the thermo groups to the given `path` on disk, where `path`
        points to the top-level folder of the thermo groups.
        """
        if not os.path.exists(path): os.mkdir(path)
        self.groups['group'].save(os.path.join(path, 'group.py'))
        self.groups['gauche'].save(os.path.join(path, 'gauche.py'))
        self.groups['int15'].save(os.path.join(path, 'int15.py'))
        self.groups['ring'].save(os.path.join(path, 'ring.py'))
        self.groups['radical'].save(os.path.join(path, 'radical.py'))
        self.groups['polycyclic'].save(os.path.join(path, 'polycyclic.py'))        
        self.groups['other'].save(os.path.join(path, 'other.py'))

    def loadOld(self, path):
        """
        Load the old RMG thermo database from the given `path` on disk, where
        `path` points to the top-level folder of the old RMG database.
        """
        # The old database does not have a depository, so create an empty one
        self.depository = {}
        self.depository['stable']  = ThermoDepository(label='stable', name='Stable Molecules')
        self.depository['radical'] = ThermoDepository(label='radical', name='Radical Molecules')
        
        for (root, dirs, files) in os.walk(os.path.join(path, 'thermo_libraries')):
            if os.path.exists(os.path.join(root, 'Dictionary.txt')) and os.path.exists(os.path.join(root, 'Library.txt')):
                library = ThermoLibrary(label=os.path.basename(root), name=os.path.basename(root))
                library.loadOld(
                    dictstr = os.path.join(root, 'Dictionary.txt'),
                    treestr = '',
                    libstr = os.path.join(root, 'Library.txt'),
                    numParameters = 12,
                    numLabels = 1,
                    pattern = False,
                )
                library.label = os.path.basename(root)
                self.libraries[library.label] = library

        self.groups = {}
        self.groups['group'] = ThermoGroups(label='group', name='Functional Group Additivity Values').loadOld(
            dictstr = os.path.join(path, 'thermo_groups', 'Group_Dictionary.txt'),
            treestr = os.path.join(path, 'thermo_groups', 'Group_Tree.txt'),
            libstr = os.path.join(path, 'thermo_groups', 'Group_Library.txt'),
            numParameters = 12,
            numLabels = 1,
            pattern = True,
        )
        self.groups['gauche'] = ThermoGroups(label='gauche', name='Gauche Interaction Corrections').loadOld(
            dictstr = os.path.join(path, 'thermo_groups', 'Gauche_Dictionary.txt'),
            treestr = os.path.join(path, 'thermo_groups', 'Gauche_Tree.txt'),
            libstr = os.path.join(path, 'thermo_groups', 'Gauche_Library.txt'),
            numParameters = 12,
            numLabels = 1,
            pattern = True,
        )
        self.groups['int15'] = ThermoGroups(label='int15', name='1,5-Interaction Corrections').loadOld(
            dictstr = os.path.join(path, 'thermo_groups', '15_Dictionary.txt'),
            treestr = os.path.join(path, 'thermo_groups', '15_Tree.txt'),
            libstr = os.path.join(path, 'thermo_groups', '15_Library.txt'),
            numParameters = 12,
            numLabels = 1,
            pattern = True,
        )
        self.groups['radical'] = ThermoGroups(label='radical', name='Radical Corrections').loadOld(
            dictstr = os.path.join(path, 'thermo_groups', 'Radical_Dictionary.txt'),
            treestr = os.path.join(path, 'thermo_groups', 'Radical_Tree.txt'),
            libstr = os.path.join(path, 'thermo_groups', 'Radical_Library.txt'),
            numParameters = 12,
            numLabels = 1,
            pattern = True,
        )
        self.groups['ring'] = ThermoGroups(label='ring', name='Ring Corrections').loadOld(
            dictstr = os.path.join(path, 'thermo_groups', 'Ring_Dictionary.txt'),
            treestr = os.path.join(path, 'thermo_groups', 'Ring_Tree.txt'),
            libstr = os.path.join(path, 'thermo_groups', 'Ring_Library.txt'),
            numParameters = 12,
            numLabels = 1,
            pattern = True,
        )
        self.groups['polycyclic'] = ThermoGroups(label='other', name='Polycyclic Ring Corrections').loadOld(
            dictstr = os.path.join(path, 'thermo_groups', 'Polycyclic_Dictionary.txt'),
            treestr = os.path.join(path, 'thermo_groups', 'Polycyclic_Tree.txt'),
            libstr = os.path.join(path, 'thermo_groups', 'Polycyclic_Library.txt'),
            numParameters = 12,
            numLabels = 1,
            pattern = True,
        )
        self.groups['other'] = ThermoGroups(label='other', name='Other Corrections').loadOld(
            dictstr = os.path.join(path, 'thermo_groups', 'Other_Dictionary.txt'),
            treestr = os.path.join(path, 'thermo_groups', 'Other_Tree.txt'),
            libstr = os.path.join(path, 'thermo_groups', 'Other_Library.txt'),
            numParameters = 12,
            numLabels = 1,
            pattern = True,
        )
        
    def pruneHeteroatoms(self, allowed=['C','H','O','S']):
        """
        Remove all species from thermo libraries that contain atoms other than those allowed.
        
        This is useful before saving the database for use in RMG-Java
        """
        allowedElements = [rmgpy.molecule.element.getElement(label) for label in allowed]
        for library in self.libraries.values():
            logging.info("Removing hetoroatoms from thermo library '{0}'".format(library.name))
            toDelete = []
            for entry in library.entries.values():
                for atom in entry.item.atoms:
                    if atom.element not in allowedElements:
                        toDelete.append(entry.label)
                        break
            for label in toDelete:
                logging.info(" {0}".format(label))
                library.entries.pop(label)

    def saveOld(self, path):
        """
        Save the old RMG thermo database to the given `path` on disk, where
        `path` points to the top-level folder of the old RMG database.
        """

        # Depository not used in old database, so it is not saved

        librariesPath = os.path.join(path, 'thermo_libraries')
        if not os.path.exists(librariesPath): os.mkdir(librariesPath)
        for library in self.libraries.values():
            libraryPath = os.path.join(librariesPath, library.label)
            if not os.path.exists(libraryPath): os.mkdir(libraryPath)
            library.saveOld(
                dictstr = os.path.join(libraryPath, 'Dictionary.txt'),
                treestr = '',
                libstr = os.path.join(libraryPath, 'Library.txt'),
            )

        groupsPath = os.path.join(path, 'thermo_groups')
        if not os.path.exists(groupsPath): os.mkdir(groupsPath)
        self.groups['group'].saveOld(
            dictstr = os.path.join(groupsPath, 'Group_Dictionary.txt'),
            treestr = os.path.join(groupsPath, 'Group_Tree.txt'),
            libstr = os.path.join(groupsPath, 'Group_Library.txt'),
        )
        self.groups['gauche'].saveOld(
            dictstr = os.path.join(groupsPath, 'Gauche_Dictionary.txt'),
            treestr = os.path.join(groupsPath, 'Gauche_Tree.txt'),
            libstr = os.path.join(groupsPath, 'Gauche_Library.txt'),
        )
        self.groups['int15'].saveOld(
            dictstr = os.path.join(groupsPath, '15_Dictionary.txt'),
            treestr = os.path.join(groupsPath, '15_Tree.txt'),
            libstr = os.path.join(groupsPath, '15_Library.txt'),
        )
        self.groups['radical'].saveOld(
            dictstr = os.path.join(groupsPath, 'Radical_Dictionary.txt'),
            treestr = os.path.join(groupsPath, 'Radical_Tree.txt'),
            libstr = os.path.join(groupsPath, 'Radical_Library.txt'),
        )
        self.groups['ring'].saveOld(
            dictstr = os.path.join(groupsPath, 'Ring_Dictionary.txt'),
            treestr = os.path.join(groupsPath, 'Ring_Tree.txt'),
            libstr = os.path.join(groupsPath, 'Ring_Library.txt'),
        )
        self.groups['polycyclic'].saveOld(
            dictstr = os.path.join(groupsPath, 'Polycyclic_Dictionary.txt'),
            treestr = os.path.join(groupsPath, 'Polycyclic_Tree.txt'),
            libstr = os.path.join(groupsPath, 'Polycyclic_Library.txt'),
        )
        self.groups['other'].saveOld(
            dictstr = os.path.join(groupsPath, 'Other_Dictionary.txt'),
            treestr = os.path.join(groupsPath, 'Other_Tree.txt'),
            libstr = os.path.join(groupsPath, 'Other_Library.txt'),
        )


    def getThermoData(self, species):
        """
        Return the thermodynamic parameters for a given :class:`Species`
        object `species`. This function first searches the loaded libraries
        in order, returning the first match found, before falling back to
        estimation via group additivity.
        
        Returns: ThermoData
        """
        # Check the libraries in order first; return the first successful match
        thermoData = self.getThermoDataFromLibraries(species)
        if thermoData is not None:
            assert len(thermoData)==3, "thermoData should be a tuple at this point, eg. (thermoData, library, entry)"
            thermoData = thermoData[0]
        else:
            # Thermo not found in any loaded libraries, so estimate
            thermoData = self.getThermoDataFromGroups(species)

        # Add Cp0 and CpInf values, if it's a ThermoData type (as opposed to eg. NASA), whether from Library or Groups
        if isinstance(thermoData, ThermoData):
            Cp0 = species.calculateCp0()
            CpInf = species.calculateCpInf()
            thermoData.Cp0 = (Cp0,"J/(mol*K)")
            thermoData.CpInf = (CpInf,"J/(mol*K)")
        # Return the resulting thermo parameters
        return thermoData
    
        
    def getThermoDataFromLibraries(self, species):
        """
        Return the thermodynamic parameters for a given :class:`Species`
        object `species`. This function first searches the loaded libraries
        in order, returning the first match found, before failing and returning None.
        
        Returns: ThermoData or None
        """
        thermoData = None
        # Check the libraries in order first; return the first successful match
        for label in self.libraryOrder:
            thermoData = self.getThermoDataFromLibrary(species, self.libraries[label])
            if thermoData is not None:
                assert len(thermoData) == 3, "thermoData should be a tuple at this point"
                thermoData[0].comment += 'Thermo library: ' + label
                return thermoData
        return None
    
    def findCp0andCpInf(self, species, thermoData):
        """
        Calculate the Cp0 and CpInf values, and add them to the thermoData object.
        
        Modifies thermoData in place and doesn't return anything
        """
        if not isinstance(thermoData,ThermoData):
            return # Just skip it
            raise Exception("Trying to add Cp0 to something that's not a ThermoData: {0!r}".format(thermoData))
        Cp0 = species.calculateCp0()
        CpInf = species.calculateCpInf()  
        thermoData.Cp0 = (Cp0,"J/(mol*K)")
        thermoData.CpInf = (CpInf,"J/(mol*K)")
                
                
    def getAllThermoData(self, species):
        """
        Return all possible sets of thermodynamic parameters for a given
        :class:`Species` object `species`. The hits from the depository come
        first, then the libraries (in order), and then the group additivity
        estimate. This method is useful for a generic search job.
        
        Returns: a list of tuples (ThermoData, source, entry) 
        (Source is a library or depository, or None)
        """
        thermoDataList = []
        # Data from depository comes first
        thermoDataList.extend(self.getThermoDataFromDepository(species))
        # Data from libraries comes second
        for label in self.libraryOrder:
            data = self.getThermoDataFromLibrary(species, self.libraries[label])
            if data: 
                assert len(data) == 3, "thermoData should be a tuple at this point"
                data[0].comment += label
                thermoDataList.append(data)
        # Last entry is always the estimate from group additivity
        # Make it a tuple
        data = (self.getThermoDataFromGroups(species), None, None)
        thermoDataList.append(data)

        # Return all of the resulting thermo parameters
        return thermoDataList

    def getThermoDataFromDepository(self, species):
        """
        Return all possible sets of thermodynamic parameters for a given
        :class:`Species` object `species` from the depository. If no
        depository is loaded, a :class:`DatabaseError` is raised.
        
        Returns: a list of tuples (thermoData, depository, entry) without any Cp0 or CpInf data.
        """
        items = []
        for label, entry in self.depository['stable'].entries.iteritems():
            for molecule in species.molecule:
                if molecule.isIsomorphic(entry.item):
                    items.append((deepcopy(entry.data), self.depository['stable'], entry))
                    break
        for label, entry in self.depository['radical'].entries.iteritems():
            for molecule in species.molecule:
                if molecule.isIsomorphic(entry.item):
                    items.append((deepcopy(entry.data), self.depository['radical'], entry))
                    break
        return items

    def getThermoDataFromLibrary(self, species, library):
        """
        Return the set of thermodynamic parameters corresponding to a given
        :class:`Species` object `species` from the specified thermodynamics
        `library`. If `library` is a string, the list of libraries is searched
        for a library with that name. If no match is found in that library,
        ``None`` is returned. If no corresponding library is found, a
        :class:`DatabaseError` is raised.
        
        Returns a tuple: (ThermoData, library, entry)  or None.
        """
        for label, entry in library.entries.iteritems():
            for molecule in species.molecule:
                if molecule.isIsomorphic(entry.item) and entry.data is not None:
                    thermoData = deepcopy(entry.data)
                    self.findCp0andCpInf(species, thermoData)
                    return (thermoData, library, entry)
        return None

    def getThermoDataFromGroups(self, species):
        """
        Return the set of thermodynamic parameters corresponding to a given
        :class:`Species` object `species` by estimation using the group
        additivity values. If no group additivity values are loaded, a
        :class:`DatabaseError` is raised.
        
        The resonance isomer (molecule) with the lowest H298 is used, and as a side-effect
        the resonance isomers (items in `species.molecule` list) are sorted in ascending order.
        
        Returns: ThermoData
        """       
        thermo = []
        for molecule in species.molecule:
            molecule.clearLabeledAtoms()
            molecule.updateAtomTypes()
            tdata = self.estimateThermoViaGroupAdditivity(molecule)
            thermo.append(tdata)

        H298 = numpy.array([t.getEnthalpy(298.) for t in thermo])
        indices = H298.argsort()
        
        species.molecule = [species.molecule[ind] for ind in indices]
        
        thermoData = thermo[indices[0]]
        self.findCp0andCpInf(species, thermoData)
        
        return thermoData
        
    def estimateRadicalThermoViaHBI(self, molecule, stableThermoEstimator ):
        """
        Estimate the thermodynamics of a radical by saturating it,
        applying the provided stableThermoEstimator method on the saturated species,
        then applying hydrogen bond increment corrections for the radical
        site(s) and correcting for the symmetry.
        """
        assert molecule.getRadicalCount() > 0, "Method only valid for radicals."
        
        # Make a copy of the structure so we don't change the original
        saturatedStruct = molecule.copy(deep=True)
        
        # Saturate structure by replacing all radicals with bonds to
        # hydrogen atoms
        added = {}
        for atom in saturatedStruct.atoms:
            for i in range(atom.radicalElectrons):
                H = Atom('H')
                bond = Bond(atom, H, 'S')
                saturatedStruct.addAtom(H)
                saturatedStruct.addBond(bond)
                if atom not in added:
                    added[atom] = []
                added[atom].append([H, bond])
                atom.decrementRadical()
        
        # Update the atom types of the saturated structure (not sure why
        # this is necessary, because saturating with H shouldn't be
        # changing atom types, but it doesn't hurt anything and is not
        # very expensive, so will do it anyway)
        saturatedStruct.updateConnectivityValues()
        saturatedStruct.sortVertices()
        saturatedStruct.updateAtomTypes()
        
        # Get thermo estimate for saturated form of structure
        thermoData = stableThermoEstimator(saturatedStruct)
        if thermoData is None:
            logging.info("Thermo data of saturated {0} of molecule {1} is None.".format(saturatedStruct, molecule))
            return None
        assert thermoData is not None, "Thermo data of saturated {0} of molecule {1} is None!".format(saturatedStruct, molecule)
        
        # Undo symmetry number correction for saturated structure
        saturatedStruct.calculateSymmetryNumber()
        thermoData.S298.value_si += constants.R * math.log(saturatedStruct.symmetryNumber)
        # Correct entropy for symmetry number of radical structure
        molecule.calculateSymmetryNumber()
        thermoData.S298.value_si -= constants.R * math.log(molecule.symmetryNumber)
        
        # For each radical site, get radical correction
        # Only one radical site should be considered at a time; all others
        # should be saturated with hydrogen atoms
        for atom in added:
            # Remove the added hydrogen atoms and bond and restore the radical
            for H, bond in added[atom]:
                saturatedStruct.removeBond(bond)
                saturatedStruct.removeAtom(H)
                atom.incrementRadical()
            saturatedStruct.updateConnectivityValues()
            try:
                self.__addGroupThermoData(thermoData, self.groups['radical'], saturatedStruct, {'*':atom})
            except KeyError:
                logging.error("Couldn't find in radical thermo database:")
                logging.error(molecule)
                logging.error(molecule.toAdjacencyList())
                raise
            # Re-saturate
            for H, bond in added[atom]:
                saturatedStruct.addAtom(H)
                saturatedStruct.addBond(bond)
                atom.decrementRadical()
            # Subtract the enthalpy of the added hydrogens
            for H, bond in added[atom]:
                thermoData.H298.value_si -= 52.103 * 4184

        return thermoData
        
        
    def estimateThermoViaGroupAdditivity(self, molecule):
        """
        Return the set of thermodynamic parameters corresponding to a given
        :class:`Molecule` object `molecule` by estimation using the group
        additivity values. If no group additivity values are loaded, a
        :class:`DatabaseError` is raised.
        """
        # For thermo estimation we need the atoms to already be sorted because we
        # iterate over them; if the order changes during the iteration then we
        # will probably not visit the right atoms, and so will get the thermo wrong
        molecule.sortVertices()

        # Create the ThermoData object
        thermoData = ThermoData(
            Tdata = ([300,400,500,600,800,1000,1500],"K"),
            Cpdata = ([0.0,0.0,0.0,0.0,0.0,0.0,0.0],"J/(mol*K)"),
            H298 = (0.0,"kJ/mol"),
            S298 = (0.0,"J/(mol*K)"),
        )

        if molecule.getRadicalCount() > 0: # radical species
            return self.estimateRadicalThermoViaHBI(molecule, self.estimateThermoViaGroupAdditivity )

        else: # non-radical species
            HNO_flag = True
            
            formula = molecule.getFormula()
            if formula in ['He', 'Ar', 'N2', 'Ne']:
                HNO_flag = False
            
            for atom in molecule.atoms:
                if atom.isCarbon() or atom.isSulfur():
                    HNO_flag = False
            
            # Generate estimate of thermodynamics
            if HNO_flag:
                for atom in molecule.atoms:
                    # Iterate over heavy (non-hydrogen) atoms
                    if atom.isNonHydrogen():
                        # Get initial thermo estimate from main group database
                        try:
                            self.__addGroupThermoData(thermoData, self.groups['HNO'], molecule, {'*':atom})
                        except KeyError:
                            logging.error("Couldn't find in HNO thermo database:")
                            logging.error(molecule)
                            logging.error(molecule.toAdjacencyList())
                            raise
            else:
                for atom in molecule.atoms:
                    # Iterate over heavy (non-hydrogen) atoms
                    if atom.isNonHydrogen():
                        # Get initial thermo estimate from main group database
                        try:
                            self.__addGroupThermoData(thermoData, self.groups['group'], molecule, {'*':atom})
                        except KeyError:
                            logging.error("Couldn't find in main thermo database:")
                            logging.error(molecule)
                            logging.error(molecule.toAdjacencyList())
                            raise
                        # Correct for gauche and 1,5- interactions
                        try:
                            self.__addGroupThermoData(thermoData, self.groups['gauche'], molecule, {'*':atom})
                        except KeyError: pass
                        try:
                            self.__addGroupThermoData(thermoData, self.groups['int15'], molecule, {'*':atom})
                        except KeyError: pass
                        try:
                            self.__addGroupThermoData(thermoData, self.groups['other'], molecule, {'*':atom})
                        except KeyError: pass

                # Do ring corrections separately because we only want to match
                # each ring one time
                
                if molecule.isCyclic():                
                    if molecule.getAllPolycyclicVertices():
                        # If the molecule has fused ring atoms, this implies that we are dealing
                        # with a polycyclic ring system, for which separate ring strain corrections may not
                        # be adequate.  Therefore, we search the polycyclic thermo group corrections
                        # instead of adding single ring strain corrections within the molecule.
                        # For now, assume only one  polycyclic RSC can be found per molecule
                        try:
                            self.__addGroupThermoData(thermoData, self.groups['polycyclic'], molecule, {})
                        except:
                            logging.error("Couldn't find in polycyclic ring database:")
                            logging.error(molecule)
                            logging.error(molecule.toAdjacencyList())
                            raise
                    else:
                        rings = molecule.getSmallestSetOfSmallestRings()
                        for ring in rings:
                            # Make a temporary structure containing only the atoms in the ring
                            # NB. if any of the ring corrections depend on ligands not in the ring, they will not be found!
                            ringCorrection = None
                            for atom in ring:
                            
                                try:
                                    correction = self.__addGroupThermoData(None, self.groups['ring'], molecule, {'*':atom})
                                except KeyError:
                                    logging.error("Couldn't find in ring database:")
                                    logging.error(ring)
                                    logging.error(ring.toAdjacencyList())
                                    raise
                        
                                if ringCorrection is None or ringCorrection.H298.value_si < correction.H298.value_si:
                                    ringCorrection = correction
                        
                        self.__addThermoData(thermoData, ringCorrection)
                
        # Correct entropy for symmetry number
        molecule.calculateSymmetryNumber()
        thermoData.S298.value_si -= constants.R * math.log(molecule.symmetryNumber)

        return thermoData

    def __addThermoData(self, thermoData1, thermoData2):
        """
        Add the thermodynamic data `thermoData2` to the data `thermoData1`,
        and return `thermoData1`.
        """
        if thermoData1 is None:
            return thermoData2
        else:
            if len(thermoData1.Tdata.value_si) != len(thermoData2.Tdata.value_si) or any([T1 != T2 for T1, T2 in zip(thermoData1.Tdata.value_si, thermoData2.Tdata.value_si)]):
                raise Exception('Cannot add these ThermoData objects due to their having different temperature points.')
        
            for i in range(thermoData1.Tdata.value_si.shape[0]):
                thermoData1.Cpdata.value_si[i] += thermoData2.Cpdata.value_si[i]
            thermoData1.H298.value_si += thermoData2.H298.value_si
            thermoData1.S298.value_si += thermoData2.S298.value_si

            if thermoData1.comment:
                thermoData1.comment += ' + {0}'.format(thermoData2.comment)
            else:
                thermoData1.comment = 'Thermo group additivity estimation: ' + thermoData2.comment
        
            return thermoData1

    def __addGroupThermoData(self, thermoData, database, molecule, atom):
        """
        Determine the group additivity thermodynamic data for the atom `atom`
        in the structure `structure`, and add it to the existing thermo data
        `thermoData`.
        """
        
        node0 = []
        
        if database == self.groups['HNO']:
            node0.append(database.descendTree(molecule, atom, None, 2))
            node_3 = database.descendTree(molecule, atom, None, 3)
            if node_3 is not None:
                node0.append(node_3)
        else:
            node0.append(database.descendTree(molecule, atom, None))
        
        if None in node0:
            raise KeyError('Node not found in database.')

        # It's possible (and allowed) that items in the tree may not be in the
        # library, in which case we need to fall up the tree until we find an
        # ancestor that has an entry in the library
        
        for node in node0:
            while node.data is None and node is not None:
                node = node.parent
            if node is None:
                raise DatabaseError('Unable to determine thermo parameters for {0}: no library entries for {1} or any of its ancestors.'.format(molecule, node0) )

            data = node.data; comment = node.label
            while isinstance(data, basestring) and data is not None:
                for entry in database.entries.values():
                    if entry.label == data:
                        data = entry.data
                        comment = entry.label
                        break
            data.comment = '{0}({1})'.format(database.label, comment)

            # This code prints the hierarchy of the found node; useful for debugging
            #result = ''
            #while node is not None:
            #   result = ' -> ' + node + result
            #   node = database.tree.parent[node]
            #print result[4:]

            thermoData = self.__addThermoData(thermoData, data)
            
        return thermoData

    def addThermoGroupsHNOFromTrainingSet(self,path):
        """
        Determine the group additivity of HNO
        """
        
        import numpy as np
        from scipy.optimize import minimize
        import matplotlib.pyplot as plt
        import pylab
        from rmgpy.molecule.group import GroupAtom, GroupBond, Group
        from rmgpy.species  import Species
        
        def thermoGroupFit(thermoGroups,*args):
            
            thermoGroups = np.reshape(thermoGroups,(A.shape[1],9))
            
            thermoDelta = np.sum(np.square(np.dot(A,thermoGroups) - thermo))
            
            return thermoDelta
        
        def thermoGroupFitLOOT(thermoGroups_LOOT,*args):
            
            thermoGroups_LOOT = np.reshape(thermoGroups_LOOT,(A_LOOT.shape[1],9))
            
            thermoDelta_LOOT = np.sum(np.square(np.dot(A_LOOT,thermoGroups_LOOT) - thermo_LOOT))
            
            return thermoDelta_LOOT
        
        
        
        n_groups = len(self.groups['HNO'].getEntriesToSave())
        
        for label, entry in self.depository['HNO'].entries.iteritems():
            
            entry.item.sortVertices()

            for atom1 in entry.item.atoms:
                if atom1.isNonHydrogen():
                    
                    ###################
                    # Find new groups #
                    ###################
                    
                    atomsGroup = []
                    atomsFound = {}
                    bondsFound = {}
                    
                    atomType          = [atom1.atomType]
                    radicalElectrons  = [atom1.radicalElectrons]
                    spinMultiplicity  = [atom1.spinMultiplicity]
                    charge            = [atom1.charge]
                    label             = '*'
                    lonePairElectrons = [atom1.lonePairs]
                    
                    atomg1 = GroupAtom(atomType, radicalElectrons, spinMultiplicity, charge, label, lonePairElectrons)
                    
                    # Add the atom to the list
                    atomsGroup.append(atomg1)
                    atomsFound[atom1] = atomg1
                    
                    for atom2, bond12 in atom1.edges.items():
                        
                        if atom2 not in atomsFound.keys():
                            
                            atomType          = [atom2.atomType]
                            radicalElectrons  = [atom2.radicalElectrons]
                            spinMultiplicity  = [atom2.spinMultiplicity]
                            charge            = [atom2.charge]
                            label             = atom2.label
                            lonePairElectrons = [atom2.lonePairs]
                    
                            atomg2 = GroupAtom(atomType, radicalElectrons, spinMultiplicity, charge, label, lonePairElectrons)
                    
                            # Add the atom to the list
                            atomsGroup.append(atomg2)
                            atomsFound[atom2] = atomg2
                            
                        if bond12 not in bondsFound.keys():
                                                
                            # Process list of bonds
                            order = [bond12.order]
                            bondg = GroupBond(atomg1, atomg2, order)
                            bondsFound[bond12] = bondg
                        
                            atomsFound[atom1].edges[atomsFound[atom2]] = bondg
                            atomsFound[atom2].edges[atomsFound[atom1]] = bondg
                            
                        
                        for atom3, bond23 in atom2.edges.items():
                            
                            if atom3 not in atomsFound.keys():
                                
                                atomType          = [atom3.atomType]
                                radicalElectrons  = [atom3.radicalElectrons]
                                spinMultiplicity  = [atom3.spinMultiplicity]
                                charge            = [atom3.charge]
                                label             = atom3.label
                                lonePairElectrons = [atom3.lonePairs]
                    
                                atomg3 = GroupAtom(atomType, radicalElectrons, spinMultiplicity, charge, label, lonePairElectrons)
                    
                                # Add the atom to the list
                                atomsGroup.append(atomg3)
                                atomsFound[atom3] = atomg3
                                
                            if bond23 not in bondsFound.keys():
                    
                                # Process list of bonds
                                order = [bond23.order]
                                bondg = GroupBond(atomg2, atomg3, order)
                                bondsFound[bond23] = bondg
                        
                                atomsFound[atom2].edges[atomsFound[atom3]] = bondg
                                atomsFound[atom3].edges[atomsFound[atom2]] = bondg
                                
                                
                        
                    group = Group(atomsGroup)    
                    group.updateConnectivityValues()
                    group.updateFingerprint()
                    
                    flagAddGroup = True
                    
                    for ifLabel, ifEntry in self.groups['HNO'].entries.iteritems():
                        if group.isIsomorphic(ifEntry.item):
                            flagAddGroup = False
                    
                    if flagAddGroup:
                        
                        ### make groups ###
                        
                        # search closest node for atom1
                        node0 = self.groups['HNO'].descendTree(entry.item, {'*':atom1}, None)
                        if node0 is None:
                            raise KeyError('Node not found in database.')

                        # It's possible (and allowed) that items in the tree may not be in the
                        # library, in which case we need to fall up the tree until we find an
                        # ancestor that has an entry in the library
                        node = node0

                        while node.data is None and node is not None:
                            node = node.parent
                        if node is None:
                            raise DatabaseError('Unable to determine thermo parameters for {0}: no library entries for {1} or any of its ancestors.'.format(molecule, node0) )
                        
                        entryLabel = group.toAdjacencyList()
                        entryLabel = entryLabel.replace("\n","")
                        entryLabel = entryLabel.replace(" ","")

                        index = n_groups
                    
                        n_groups = n_groups + 1
    
                        label = entryLabel
                        item = group
                        thermoData = ThermoData(
                                                Tdata = ([300,400,500,600,800,1000,1500],"K"),
                                                Cpdata = ([0.0,0.0,0.0,0.0,0.0,0.0,0.0],"cal/(mol*K)"),
                                                H298 = (0.0,"kcal/mol"),
                                                S298 = (0.0,"cal/(mol*K)"),
                                                )
                        reference = ''
                        referenceType = ''
                        shortDesc = ''
                        longDesc = ''
                        history = []
                            
                        self.groups['HNO'].entries[label] = Entry(
                            index = index,
                            label = label,
                            item = item,
                            data = thermoData,
                            reference = reference,
                            referenceType = referenceType,
                            shortDesc = shortDesc,
                            longDesc = longDesc.strip(),
                            history = history or [],
                        )
                    
                        self.groups['HNO'].entries[label].parent = node
                        node.children.append(self.groups['HNO'].entries[label])
                    
# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        
        A = [] 
        thermo = []
        itemCount = 0
        
        
        despositoryLabels = {}
        despositoryIndexes = {}
        
        for label, entry in self.depository['HNO'].entries.iteritems():
            
            itemCount = itemCount + 1
            
            despositoryLabels[str(itemCount-1)] = label
            despositoryIndexes[str(itemCount-1)] = entry.index
            
            thermoLocal = np.zeros(9)
            
            thermoLocal[0] = entry.data.H298.value
            thermoLocal[1] = entry.data.S298.value
            thermoLocal[2:10] = entry.data.Cpdata.value[0:7]
            
            if thermo == []:
                thermo = thermoLocal
            else:
                thermo = np.append(thermo,thermoLocal,0)
                
            # For thermo estimation we need the atoms to already be sorted because we
            # iterate over them; if the order changes during the iteration then we
            # will probably not visit the right atoms, and so will get the thermo wrong
            entry.item.sortVertices()
            
            localA = np.zeros(n_groups)
                    
            for atom1 in entry.item.atoms:
                if atom1.isNonHydrogen():
                    
                    # Get initial thermo estimate from HNO database
                    
                    ### level 2 ###
                    
                    node0 = self.groups['HNO'].descendTree(entry.item, {'*':atom1}, None, 2)
                    if node0 is None:
                        raise KeyError('Node not found in database.')

                    # It's possible (and allowed) that items in the tree may not be in the
                    # library, in which case we need to fall up the tree until we find an
                    # ancestor that has an entry in the library
                    node = node0
                    while node.data is None and node is not None:
                        node = node.parent
                    if node is None:
                        raise DatabaseError('Unable to determine thermo parameters for {0}: no library entries for {1} or any of its ancestors.'.format(molecule, node0) )
                    
                    localA[node.index-1] = localA[node.index-1] + 1
                    
                    
                    ### level 3 ###
                    
                    node = self.groups['HNO'].descendTree(entry.item, {'*':atom1}, None, 3)
                    
                    if node is not None:
                        localA[node.index-1] = localA[node.index-1] + 1

            A = np.append(A,localA,0)
        
        
        ################
        # Minimization #
        ################
                
        A = np.reshape(A,(itemCount,n_groups))
        thermo = np.reshape(thermo,(itemCount,9))
        thermoGroups = np.ones((n_groups,9))*0.01
                    
        res = minimize(thermoGroupFit, thermoGroups, args=(A, thermo), method='BFGS', options={'disp': True})
        
        thermoGroups = np.reshape(res.x,(n_groups,9))
        
        groupThermo = np.dot(A,thermoGroups)
        
        ###################################
        # Minimization analysis and print #
        ###################################
        
        fig = pylab.figure()
        
        X_Hf_ideal = [min(thermo[:,0]),max(thermo[:,0])]
        Y_Hf_ideal = [min(groupThermo[:,0]),max(groupThermo[:,0])]
        Y_Hf_ideal_neg5 = [min(groupThermo[:,0])-5,max(groupThermo[:,0])-5]
        Y_Hf_ideal_pos5 = [min(groupThermo[:,0])+5,max(groupThermo[:,0])+5]
        
        pylab.subplot(3,3,1)
        pylab.plot(thermo[:,0],groupThermo[:,0],'o',X_Hf_ideal,Y_Hf_ideal,'g',X_Hf_ideal,Y_Hf_ideal_neg5,'g--',X_Hf_ideal,Y_Hf_ideal_pos5,'g--')
        pylab.title('Hf_298 (kcal/mol)')
        
        X_S_ideal = [min(thermo[:,1]),max(thermo[:,1])]
        Y_S_ideal = [min(groupThermo[:,1]),max(groupThermo[:,1])]
        Y_S_ideal_neg5 = [min(groupThermo[:,1])-5,max(groupThermo[:,1])-5]
        Y_S_ideal_pos5 = [min(groupThermo[:,1])+5,max(groupThermo[:,1])+5]
        
        pylab.subplot(3,3,2)
        pylab.plot(thermo[:,1],groupThermo[:,1],'o',X_S_ideal,Y_S_ideal,'g',X_S_ideal,Y_S_ideal_neg5,'g--',X_S_ideal,Y_S_ideal_pos5,'g--')
        pylab.title('S_298 (cal/(mol*K)')
        
        T_range = [300,400,500,600,800,1000,1200]
        for i in range(7):
            X_Cp_ideal = [min(thermo[:,2+i]),max(thermo[:,2+i])]
            Y_Cp_ideal = [min(groupThermo[:,2+i]),max(groupThermo[:,2+i])]
            Y_Cp_ideal_neg5 = [min(groupThermo[:,2+i])-5,max(groupThermo[:,2+i])-5]
            Y_Cp_ideal_pos5 = [min(groupThermo[:,2+i])+5,max(groupThermo[:,2+i])+5]
        
            pylab.subplot(3,3,3+i)
            pylab.plot(thermo[:,2+i],groupThermo[:,2+i],'o',X_Cp_ideal,Y_Cp_ideal,'g',X_Cp_ideal,Y_Cp_ideal_neg5,'g--',X_Cp_ideal,Y_Cp_ideal_pos5,'g--')
            pylab.title('Cp_'+str(T_range[i])+' cal/(mol*K)')
        
        pylab.savefig(os.path.join(path, 'thermo/Group_additivity_fit_HNO.pdf'))
        
        for label, entry in self.groups['HNO'].entries.iteritems():
            if entry.index != -1:
                entry.data.H298.value = thermoGroups[entry.index-1][0]
                entry.data.S298.value = thermoGroups[entry.index-1][1]
                entry.data.Cpdata.value = thermoGroups[entry.index-1][2:9]
        
        self.groups['HNO'].save(os.path.join(path, 'thermo/HNO_groups.py'))
        
        f = open(os.path.join(path, 'thermo/statistics.txt'), 'w')
        f.write('index occurrences\n\n')
        
        for label, entry in self.groups['HNO'].entries.iteritems():
            if entry.index != -1:
                f.write(str(entry.index) + ' ' + str(int(sum(A[:,entry.index-1]))) + '\n')
                
        f.close()
        
        errorTitles = ['Hf298','S298','Cp300','Cp400','Cp500','Cp600','Cp800','Cp1000','Cp1500']
        
        f = open(os.path.join(path, 'thermo/errors.txt'), 'w')
        
        titleCount = 0
        
        for title in errorTitles:
        
            f.write('3 largest errors in ' + title + '\n')
        
            delta = np.dot(A,thermoGroups) - thermo
        
            delta = delta[:,titleCount]
            delta_abs = abs(delta)
            delta_indexes = delta_abs.argsort()[-3:][::-1]
        
            f.write('rank label error\n')
   
            count = 0
        
            for index in delta_indexes:
                count = count + 1
                f.write(str(count) + ' '  + ' ' + despositoryLabels[str(index)] + ' ' + str(delta[index]) + ' \n')
                
                
            f.write('\n\n')
            
            titleCount = titleCount + 1
          
        f.close()
        
        ######################
        # Leave-One-Out Test #
        ######################
        
        f = open(os.path.join(path, 'thermo/LOOT.txt'), 'w')
        f.write('index label Hf298 S298 Cp300 Cp400 Cp500 Cp600 Cp800 Cp1000 Cp1500\n\n')
        f.close()
        
#        for item_no in range(itemCount):
#            
#            A_LOOT             = A
#            thermo_LOOT        = thermo #np.reshape(thermo,(itemCount,9))
#            thermoGroups_LOOT  = np.ones((n_groups,9))*0.01
#            
#            A_item = A_LOOT[item_no]
#            A_LOOT = np.delete(A_LOOT,item_no,0)
#            thermo_LOOT = np.delete(thermo_LOOT,item_no,0)
#            
#            res_LOOT = minimize(thermoGroupFitLOOT, thermoGroups_LOOT, args=(A_LOOT, thermo_LOOT), method='BFGS', options={'disp': True})
#        
#            thermoGroups_LOOT = np.reshape(res_LOOT.x,(n_groups,9))
#        
#            groupThermo_LOOT = np.dot(A_item,thermoGroups_LOOT)
#            
#            f = open(os.path.join(path, 'thermo/LOOT.txt'), 'a')
#            
#            f.write(str(despositoryIndexes[str(item_no)]) + ' ' + despositoryLabels[str(item_no)] 
#                    + ' ' + str(self.depository['HNO'].entries[despositoryLabels[str(item_no)]].data.H298.value) 
#                    + ' ' + str(self.depository['HNO'].entries[despositoryLabels[str(item_no)]].data.S298.value) 
#                    + ' ' + str(self.depository['HNO'].entries[despositoryLabels[str(item_no)]].data.Cpdata.value[0])
#                    + ' ' + str(self.depository['HNO'].entries[despositoryLabels[str(item_no)]].data.Cpdata.value[1])
#                    + ' ' + str(self.depository['HNO'].entries[despositoryLabels[str(item_no)]].data.Cpdata.value[2])
#                    + ' ' + str(self.depository['HNO'].entries[despositoryLabels[str(item_no)]].data.Cpdata.value[3])
#                    + ' ' + str(self.depository['HNO'].entries[despositoryLabels[str(item_no)]].data.Cpdata.value[4])
#                    + ' ' + str(self.depository['HNO'].entries[despositoryLabels[str(item_no)]].data.Cpdata.value[5])
#                    + ' ' + str(self.depository['HNO'].entries[despositoryLabels[str(item_no)]].data.Cpdata.value[6])
#                    + ' '
#                    + ' '
#                    + ' '
#                    + ' ' + str(groupThermo_LOOT[0])
#                    + ' ' + str(groupThermo_LOOT[1])
#                    + ' ' + str(groupThermo_LOOT[2])
#                    + ' ' + str(groupThermo_LOOT[3])
#                    + ' ' + str(groupThermo_LOOT[4])
#                    + ' ' + str(groupThermo_LOOT[5])
#                    + ' ' + str(groupThermo_LOOT[6])
#                    + ' ' + str(groupThermo_LOOT[7])
#                    + ' ' + str(groupThermo_LOOT[8])
#                    + '\n')
#            
#            f.close()
            
        

   #     res[1]
                    
        return None

            
            
