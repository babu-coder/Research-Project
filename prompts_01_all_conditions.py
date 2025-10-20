# ==========================================
# Canine Conditions prompts created for each condition
# ==========================================

canine_conditions = {
    "perihilar_infiltrate": {
        "prompt": """You are a  radiologist.
I will provide you with a report study.
Determine if there is any evidence of perihilar infiltrate,
using the following classifications: "Negative"
if there is no evidence of perihilar infiltrate;
"Positive" when perihilar infiltrate are clearly identified,
indicating issues;
Provide the response in the format:
"Negative" or "Positive" ."""
    },

    "pneumonia": {
        "prompt": """You are a  radiologist.
I will provide you with a report study.
Determine if there is any evidence of pneumonia,
 using the following classifications:
"Negative" if there is no evidence of pneumonia;
"Positive" when pneumonia are clearly identified,
indicating issues;
Provide the response in the format:
"Negative" or "Positive" ."""
    },

    "bronchitis": {
        "prompt": """You are a  radiologist.
I will provide you with a report study.
Determine if there is any evidence of bronchitis,
using the following classifications:
"Negative" if there is no evidence of bronchitis;
"Positive" when bronchitis are clearly identified,
indicating issues;
Provide the response in the format:
"Negative" or "Positive" ."""
    },

    "interstitial": {
        "prompt": """You are a  radiologist.
I will provide you with a report study.
Determine if there is any evidence of interstitial pattern,
using the following classifications:
"Negative" if there is no evidence of interstitial changes;
"Positive" when interstitial changes are clearly identified,
indicating issues;
Provide the response in the format:
"Negative" or "Positive" ."""
    },

    "diseased_lungs": {
        "prompt": """You are a  radiologist.
I will provide you with a report study.
Determine if there is any evidence of diseased lungs,
using the following classifications:
"Negative" if there is no evidence of diseased lungs;
"Positive" when diseased lungs are clearly identified,
indicating issues;
Provide the response in the format:
"Negative" or "Positive" ."""
    },

    "hypo_plastic_trachea": {
        "prompt": """You are a  radiologist.
I will provide you with a report study.
Determine if there is any evidence of hypoplastic trachea,
using the following classifications:
"Negative" if there is no evidence of hypoplastic trachea;
"Positive" when hypoplastic trachea is clearly identified,
indicating issues;
Provide the response in the format:
"Negative" or "Positive" ."""
    },

    "cardiomegaly": {
        "prompt": """You are a  radiologist.
I will provide text from radiology report.
Based on the report, determine if cardiomegaly is present.
Strictly limit your answers to "Negative" or "Positive" .
Respond "Positive" if the report mentions any of the following: enlarged heart,
increased cardiac silhouette, cardiomegaly, or increased vertebral
heart score (VHS).
Respond "Negative" if the heart size is described as normal or
if no cardiac enlargement is mentioned.
Provide the response in the format:
"Negative" or "Positive" ."""
    },

    "pulmonary_nodules": {
        "prompt": """You are a radiologist.
I will provide text from report.
Based on the report, determine if pulmonary nodules are present.
Strictly limit your answers to "Negative" or "Positive" .
Respond "Positive" if the report mentions any of the following:
 pulmonary nodule(s), rounded opacities < 3cm, multiple nodular opacities,
 or metastatic nodules.
Respond "Negative" if no pulmonary nodules are mentioned or if lungs
 are described as clear.
Provide the response in the format:
 "Negative" or "Positive" . """
    },

    "pleural_effusion": {
        "prompt": """You are a radiologist.
I will provide text from report.
Based on the report, determine if pericardial fluid or effusion is present.
Strictly limit your answers to "Negative" or "Positive" .
Respond "Positive" if the report mentions any of the following:
 pericardial effusion, fluid in the pericardial space,
 globoid cardiac silhouette, or cardiac tamponade.
Respond "Negative" if no pericardial fluid is mentioned
or if the pericardial space is described as normal.
Provide the response in the format:
 "Negative" or "Positive" ."""
    },

    "rtm": {
        "prompt": """You are a  radiologist.
I will provide you with a report study.
Determine if there is any evidence of rtm, using the following classifications:
"Negative" if there is no evidence of rtm;
"Positive" when rtm are clearly identified,
indicating issues;
Provide the response in the format:
"Negative" or "Positive" ."""
    },

    "focal_caudodorsal_lung": {
        "prompt": """You are a  radiologist.
I will provide you with a report study.
Determine if there is any evidence of focal caudodorsal lung abnormality,
using the following classifications:
"Negative" if there is no evidence of focal caudodorsal lung abnormality;
"Positive" when focal caudodorsal lung changes are clearly identified,
indicating potential clinical issues such as localized consolidation or lesion;
Provide the response in the format:
"Negative" or "Positive" ."""
    },

    "focal_perihilar": {
        "prompt": """You are a  radiologist.
I will provide you with a report study.
Determine if there is any evidence of focal perihilar abnormality,
using the following classifications:
"Negative" if there is no evidence of focal perihilar abnormality;
"Positive" when focal perihilar changes are clearly identified,
indicating potential clinical issues such as localized infiltrate,
mass, or lesion;
Provide the response in the format:
"Negative" or "Positive" ."""
    },

    "pulmonary_hypoinflation": {
        "prompt": """You are  radiologist.
I will provide text from report.
Based on the report, determine if pulmonary hypoinflation is present.
Strictly limit your answers to "Negative" or "Positive" .
Respond "Positive" if the report mentions any of the following:
decreased lung volume, hypoinflation, microcardia, elevated diaphragm,
or crowded pulmonary vessels.
Respond "Negative" if lung inflation is described as normal or
if no signs of hypoinflation are mentioned.
Provide the response in the format:
"Negative" or "Positive" ."""
    },

    "right_sided_cardiomegaly": {
        "prompt": """You are a radiologist.
I will provide text from report.
Based on the report, determine if right-sided cardiomegaly is present.
Strictly limit your answers to "Negative" or "Positive" .
Respond "Positive" if the report mentions any of the following:
right atrial enlargement, right ventricular enlargement,
right-sided heart enlargement, or reversed D-shaped cardiac silhouette.
Respond "Negative" if the right heart chambers are described as normal
or if no right-sided enlargement is mentioned.
Provide the response in the format:
"Negative" or "Positive" ."""
    },

    "pericardial_effusion": {
        "prompt": """You are a radiologist.
I will provide text from report.
Based on the report, determine if pericardial fluid or effusion is present.
Strictly limit your answers to "Negative" or "Positive" .
Respond "Positive" if the report mentions any of the following:
pericardial effusion, fluid in the pericardial space, globoid cardiac
silhouette, or cardiac tamponade.
Respond "Negative" if no pericardial fluid is mentioned or
if the pericardial space is described as normal.
Provide the response in the format:
"Negative" or "Positive" ."""
    },

    "bronchiectasis": {
        "prompt": """You are a radiologist.
I will provide text from report.
Based on the report, determine if bronchiectasis is present.
Strictly limit your answers to "Negative" or "Positive" .
Respond "Positive" if the report mentions any of the following:
bronchiectasis, dilated bronchi, bronchial dilation, or cylindrical/saccular
airway changes.
Respond "Negative" if airways are described as normal or if no
bronchial dilation is mentioned.
Provide the response in the format:
"Negative" or "Positive" ."""
    },

    "pulmonary_vessel_enlargement": {
        "prompt": """You are a radiologist.
I will provide text from report.
Based on the report, determine if pulmonary vasculature enlargement is present.
Strictly limit your answers to "Negative" or "Positive" .
Respond "Positive" if the report mentions any of the following:
enlarged pulmonary vessels, increased pulmonary vasculature,
dilated pulmonary vessels, or pulmonary vascular congestion.
Respond "Negative" if vessels are described as normal or
if no vascular abnormalities are mentioned..
Provide the response in the format:
"Negative" or "Positive" ."""
    },

    "left_sided_cardiomegaly": {
        "prompt": """You are a board certified veterinary radiologist.
I will provide text from a canine report.
Based on the report, determine if left-sided cardiomegaly is present.
Strictly limit your answers to "Negative" or "Positive" .
Respond "Positive" if the report mentions any of the following:
 left atrial enlargement, left ventricular enlargement,
 left-sided heart enlargement, or dorsally displaced trachea at the
level of the carina. Respond "Negative" if the left heart chambers are
described as normal or if no left-sided enlargement is mentioned.
Provide the response in the format:
"Negative" or "Positive"."""
    },

    "thoracic_lymphadenopathy": {
        "prompt": """You are a  radiologist.
I will provide text from a canine report.
Based on the report, determine whether thoracic lymphadenopathy is present.
Response Criteria: Respond "Positive" if the report explicitly mentions or
 clearly suggests any of the following: enlarged thoracic lymph nodes,
 sternal lymphadenopathy, tracheobronchial
 lymphadenopathy, mediastinal lymph node enlargement,
 or increased opacity in the cranial mediastinum
 consistent with lymphadenopathy.
Respond "Negative" if the report describes lymph nodes as normal or does not
 mention any finding suggesting enlargement.
Provide your response only as: "Negative", "Positive"."""
    },

    "esophagitis": {
        "prompt": """You are a  radiologist.
I will provide you with a report study.
Determine if there is any evidence of esophagitis,
using the following classifications:
"Negative" if there is no evidence of esophagitis;
"Positive" when esophagitis are clearly identified,
indicating issues;
Provide the response in the format:
 "Negative" or "Positive" ."""
    },

    "vhs_v2": {
        "prompt": """You are a  radiologist.
I will provide you with a report study.
Determine if there is any evidence of vhs, using the following classifications:
"Negative" if there is no evidence of vhs;
"Positive" when vhs are clearly identified,
 indicating issues;
Provide the response in the format:
 "Negative" or "Positive" ."""
    }
}

conditions = canine_conditions
