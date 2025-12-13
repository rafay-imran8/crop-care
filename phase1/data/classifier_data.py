# Small labeled dataset for sanity checking
samples = [
    ("yellow spots on wheat leaves", 1),     # disease
    ("brown rust appearing on wheat", 1),
    ("aphids attacking maize crop", 0),      # pest
    ("locust infestation in fields", 0),
    ("how often to irrigate rice", 2),        # irrigation
    ("best irrigation schedule for maize", 2),
    ("low nitrogen in soil", 3),              # soil
    ("soil lacks organic matter", 3),
]

label_map = {
    0: "pest",
    1: "disease",
    2: "irrigation",
    3: "soil"
}
