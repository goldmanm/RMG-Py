#!/usr/bin/env python
# encoding: utf-8

name = "seed"
shortDesc = ""
longDesc = """

"""
autoGenerated=True
entry(
    index = 0,
    label = "H + H <=> H2",
    degeneracy = 0.5,
    duplicate = True,
    kinetics = Arrhenius(A=(5.45e+10,'cm^3/(mol*s)'), n=0, Ea=(6.276,'kJ/mol'), T0=(1,'K'), Tmin=(278,'K'), Tmax=(372,'K'), comment="""Matched reaction 56 H + H <=> H2 in R_Recombination/training
    This reaction matched rate rule [Root_1R->H_N-2R-inRing_2R->H]
    family: R_Recombination"""),
    longDesc = 
"""
Matched reaction 56 H + H <=> H2 in R_Recombination/training
This reaction matched rate rule [Root_1R->H_N-2R-inRing_2R->H]
family: R_Recombination
""",
)

entry(
    index = 1,
    label = "H + H <=> H2",
    degeneracy = 0.5,
    duplicate = True,
    kinetics = Arrhenius(A=(5.45e+10,'cm^3/(mol*s)'), n=0, Ea=(6.276,'kJ/mol'), T0=(1,'K'), Tmin=(278,'K'), Tmax=(372,'K'), comment="""Matched reaction 56 H + H <=> H2 in R_Recombination/training
    This reaction matched rate rule [Root_1R->H_N-2R-inRing_2R->H]
    family: R_Recombination"""),
    longDesc = 
"""
Matched reaction 56 H + H <=> H2 in R_Recombination/training
This reaction matched rate rule [Root_1R->H_N-2R-inRing_2R->H]
family: R_Recombination
""",
)

entry(
    index = 2,
    label = "O2 + H2 <=> H + [O]O",
    degeneracy = 4.0,
    kinetics = Arrhenius(A=(2.9e+14,'cm^3/(mol*s)','*|/',5), n=0, Ea=(236.982,'kJ/mol'), T0=(1,'K'), Tmin=(300,'K'), Tmax=(800,'K'), comment="""Matched reaction 306 H2 + O2 <=> HO2_r12 + H in H_Abstraction/training
    This reaction matched rate rule [H2;O2b]
    family: H_Abstraction"""),
    longDesc = 
"""
Matched reaction 306 H2 + O2 <=> HO2_r12 + H in H_Abstraction/training
This reaction matched rate rule [H2;O2b]
family: H_Abstraction
""",
)

entry(
    index = 3,
    label = "O2 + H <=> [O]O",
    degeneracy = 2.0,
    kinetics = Arrhenius(A=(8.79e+10,'cm^3/(mol*s)'), n=1, Ea=(1.8828,'kJ/mol'), T0=(1,'K'), Tmin=(298,'K'), Tmax=(6000,'K'), comment="""Matched reaction 104 O2 + H <=> HO2-2 in R_Recombination/training
    This reaction matched rate rule [Root_1R->H_N-2R-inRing_N-2R->H_N-2CNOS->S_N-2CNO->C_Ext-2NO-R_N-2NO->N_N-3R!H->C]
    family: R_Recombination"""),
    longDesc = 
"""
Matched reaction 104 O2 + H <=> HO2-2 in R_Recombination/training
This reaction matched rate rule [Root_1R->H_N-2R-inRing_N-2R->H_N-2CNOS->S_N-2CNO->C_Ext-2NO-R_N-2NO->N_N-3R!H->C]
family: R_Recombination
""",
)

entry(
    index = 4,
    label = "H + OO <=> [O]O + H2",
    degeneracy = 2.0,
    kinetics = Arrhenius(A=(1.23502,'m^3/(mol*s)'), n=1.634, Ea=(25.4634,'kJ/mol'), T0=(1,'K'), comment="""Estimated using average of templates [O/H/NonDeO;H_rad] + [H2O2;Y_rad] for rate rule [H2O2;H_rad]
    Euclidian distance = 1.0
    Multiplied by reaction path degeneracy 2.0
    family: H_Abstraction"""),
    longDesc = 
"""
Estimated using average of templates [O/H/NonDeO;H_rad] + [H2O2;Y_rad] for rate rule [H2O2;H_rad]
Euclidian distance = 1.0
Multiplied by reaction path degeneracy 2.0
family: H_Abstraction
""",
)

entry(
    index = 5,
    label = "H + [O]O <=> OO",
    degeneracy = 1.0,
    kinetics = Arrhenius(A=(5250.69,'m^3/(mol*s)'), n=1.27262, Ea=(0,'kJ/mol'), T0=(1,'K'), Tmin=(300,'K'), Tmax=(1500,'K'), uncertainty=RateUncertainty(mu=0.0, var=33.1368631905, Tref=1000.0, N=1, correlation='Root_1R->H_N-2R-inRing_N-2R->H_N-2CNOS->S_N-2CNO->C_Ext-2NO-R_N-2NO->N_N-3R!H->C',), comment="""BM rule fitted to 2 training reactions at node Root_1R->H_N-2R-inRing_N-2R->H_N-2CNOS->S_N-2CNO->C_Ext-2NO-R_N-2NO->N_N-3R!H->C
        Total Standard Deviation in ln(k): 11.5401827615
    Exact match found for rate rule [Root_1R->H_N-2R-inRing_N-2R->H_N-2CNOS->S_N-2CNO->C_Ext-2NO-R_N-2NO->N_N-3R!H->C]
    Euclidian distance = 0
    family: R_Recombination"""),
    longDesc = 
"""
BM rule fitted to 2 training reactions at node Root_1R->H_N-2R-inRing_N-2R->H_N-2CNOS->S_N-2CNO->C_Ext-2NO-R_N-2NO->N_N-3R!H->C
    Total Standard Deviation in ln(k): 11.5401827615
Exact match found for rate rule [Root_1R->H_N-2R-inRing_N-2R->H_N-2CNOS->S_N-2CNO->C_Ext-2NO-R_N-2NO->N_N-3R!H->C]
Euclidian distance = 0
family: R_Recombination
""",
)

entry(
    index = 6,
    label = "[O]O + [O]O <=> O2 + OO",
    degeneracy = 1.0,
    kinetics = Arrhenius(A=(1.75e+10,'cm^3/(mol*s)'), n=0, Ea=(-13.7026,'kJ/mol'), T0=(1,'K'), Tmin=(300,'K'), Tmax=(1500,'K'), comment="""Matched reaction 405 HO2_r3 + HO2_r12 <=> H2O2 + O2 in H_Abstraction/training
    This reaction matched rate rule [Orad_O_H;O_rad/NonDeO]
    family: H_Abstraction"""),
    longDesc = 
"""
Matched reaction 405 HO2_r3 + HO2_r12 <=> H2O2 + O2 in H_Abstraction/training
This reaction matched rate rule [Orad_O_H;O_rad/NonDeO]
family: H_Abstraction
""",
)

