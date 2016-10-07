##Documentation

###Background
**MIMC_SQLITE.py**: MIMIC III Database building scripts using python SQLite3 package. 
- Contains functions for building UFM tables from MIMIC III.
- MIMIC III files were obtained from PhysioNet as per recommendation. 
- UFM tables are feature tables that lists all relevant laboratory, procedural and diagnostic events for patients of interest.
    *Each feature is labeled with prefix 'l_', 'p_', or 'd_', indicating type of feature (laboratory, procedural or diagnostic) 
    *Each feature is labled with Timestamp to facilitate temporal sequencing and context embedding
- 4 reference tables are made to assist UFM table interpretation:
    *Ref1. Demographic Information
    *Ref2. Timestamp to Epoch Integer Conversion
    *Ref3. Features Table
    *Ref4. Lab Values Lookup
- This script uses CHF readmission patients as the selection factor when making UFM feature tables.
- Also contains functions for CHF analysis.

**create_tables() and create_aux_tables()**
- This function creates the major tables ADMISSIONS, LABEVENTS, DIAGNOSES_ICD, D_ICD_DIAGNOSES, and PROCEDUREEVENTS_MV data tables. 
- Although this script uses CSV files obtained from PhysioNet to construct a flat file database image via SQLite, it is designed for future access to MIMIC files through Illidan Lab servers.
    *SQLite3 is a default python package that allows quick database building and access. However, it does not allow server use. 
    *Syntax from SQLite3 translate to MySQL for the most part. This script will be continously modified should difficulty in access arise.
- Large tables such as LABEVENTS (1.77GB) and CHARTEVENTS (>10GB) should be read in chunks
    *Runtime for this part can take up to several hours.
- Tables NOT included in 'mimic.db': _CHARTEVENTS, INPUTEVENTS, OUTPUTEVENTS, MICROBIOLOGYEVENTS, PRESCRIPTIONS, TRANSFERS_.

**get_CHF()**
- This function separates CHF patients who meet the 180-day window readmission criteria (y=1) and CHF patients who do not meet the readmission criteria (y=0).
- There are further subdivisions in the non-readmission group (e.g., discharge vs. deceased) that may be evaluated in the future.

**make_UFM()**
- This function builds the main feature table of interest, the UFM table for patients of interest.
- Querying is often split up due to SQLite3 limitations (e.g., max variables in filter cannot exceed 999 items), but these restrictions DO NOT apply in MySQL settings. 
- UFM tables are both stored as Pandas DataFrames as well as written to 'mimic.db' as SQL files. 
- Timestamps for 'd_' features: While the DIAGNOSES_ICD data table does not provide TIMESTAMPS for diagnoses, HADM_ID was used to trace each diagnosis to the patient encounter in ADMISSIONS data table where the TIMESTAMP is provided for the admission.

**make_refs()**
- This function constructs the reference tables which accompany UFM table.
- Ref1: Demographics Information
    *GENDER, Date of Birth, Date of Death (when applicable), and EXPIRE_FLAG are imported from PATIENTS data table, filtered by SUBJECT_ID.
    *INSURANCE, LANGUAGE, RELIGION, MARITAL_STATUS, and ETHNICITY are imported from ADMISSIONS data table filtered by SUBJECT_ID.
- Ref2: Timestamps to Epoch Integer Conversion
    *Each TIMESTAMP relevant to the UFM table is converted to an Epoch Integer, which can replace TIMESTAMP data in the UFM table for practical purposes.
- Ref3: Features Table
    *Each 'FEATURE' from UFM is linked to its 'ITEMID' in D_ITEMS (procedure events), D_LABITEMS (lab events), or D_ICD_DIAGNOSES (diagnosis events).
    *'TYPE' refers to 'LABEL' from D_ITEMS or D_LABITEMS which describes the concept/use represented by the feature. For diagnosis features, these are simply labeled as "DIAGNOSIS".
    *'DESCRIPTION' refers to the 'CATEGORY' descriptor in D_ITEMS or D_LABITEMS which provide 'higher level information' for the feature. e.g., 'CMP' is 'Comprehensive Metabolic Panel'.
- Ref4: Lab Values Lookup
    *For each lab event feature in UFM table, the 'VALUEOUM' value units (when applicable) and 'FLAG' (which indicates measurement beyond normal thresholds) are linked from LABEVENTS data table.
    *'LOINC REFERENCE' is linked to 'LOINC_CODE' provided in D_LABITEMS. LOINC reference codes can be looked up online (NOT provided by MIMIC), which describe in further detail about the nature of each lab procedure.
    

##FIN
