samples = [
    # ------------------
    # DISEASE (label = 1)
    # ------------------
    ("yellow spots on wheat leaves", 1),
    ("brown rust appearing on wheat", 1),
    ("orange pustules on wheat leaf surface", 1),
    ("powdery growth on maize leaves", 1),
    ("wheat leaves turning brown and dry", 1),
    ("leaf rust symptoms in wheat crop", 1),
    ("stunted maize plants with necrotic leaves", 1),
    ("maize lethal necrosis disease symptoms", 1),
    ("fungal infection spreading on wheat plants", 1),
    ("dark lesions forming on maize leaves", 1),

    # ------------------
    # PEST (label = 0)
    # ------------------
    ("aphids attacking maize crop", 0),
    ("locust infestation in fields", 0),
    ("small insects sucking sap from wheat leaves", 0),
    ("worms damaging maize roots", 0),
    ("armyworm infestation in maize field", 0),
    ("caterpillars eating wheat leaves", 0),
    ("white insects on underside of maize leaves", 0),
    ("pest attack causing holes in leaves", 0),
    ("insect larvae found in soil near maize plants", 0),
    ("termite damage in crop field", 0),

    # ------------------
    # IRRIGATION (label = 2)
    # ------------------
    ("how often to irrigate rice", 2),
    ("best irrigation schedule for maize", 2),
    ("water requirement of wheat crop", 2),
    ("irrigation frequency during wheat flowering stage", 2),
    ("over irrigation symptoms in maize", 2),
    ("when to irrigate wheat after sowing", 2),
    ("water stress signs in maize plants", 2),
    ("optimal water management for maize fields", 2),
    ("effects of drought on wheat yield", 2),
    ("drip irrigation suitability for maize", 2),

    # ------------------
    # SOIL / NUTRIENTS (label = 3)
    # ------------------
    ("low nitrogen in soil", 3),
    ("soil lacks organic matter", 3),
    ("yellowing leaves due to nutrient deficiency", 3),
    ("poor soil fertility affecting wheat yield", 3),
    ("phosphorus deficiency symptoms in maize", 3),
    ("how to improve soil structure for crops", 3),
    ("acidic soil problem in agriculture", 3),
    ("soil erosion reducing crop productivity", 3),
    ("nutrient imbalance in wheat field", 3),
    ("importance of organic manure for soil health", 3),
]
label_map = {
    0: "pest",
    1: "disease",
    2: "irrigation",
    3: "soil"
}