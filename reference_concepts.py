"""
reference_concepts.py — Predefined reference concept definitions.
Used to display concept context and compute semantic similarity.
"""

CONCEPTS = {
    "Machine Learning": (
        "Machine Learning is a subset of artificial intelligence that allows systems "
        "to learn patterns from data and improve performance without being explicitly programmed."
    ),
    "Cloud Computing": (
        "Cloud Computing is the delivery of computing services including servers, storage, "
        "databases, networking, and software over the internet to offer faster innovation, "
        "flexible resources, and economies of scale."
    ),
    "Deep Learning": (
        "Deep Learning is a branch of machine learning that uses neural networks with many "
        "layers to automatically learn representations from raw data such as images, audio, and text."
    ),
    "Natural Language Processing": (
        "Natural Language Processing is a field of AI that enables computers to understand, "
        "interpret, and generate human language in a meaningful and useful way."
    ),
    "Computer Vision": (
        "Computer Vision is a field of AI that trains computers to interpret and understand "
        "visual information from the world such as images and videos."
    ),
    "Blockchain": (
        "Blockchain is a distributed ledger technology that records transactions across multiple "
        "computers in a way that ensures the data cannot be altered retroactively."
    ),
    "Internet of Things": (
        "The Internet of Things refers to the network of physical devices embedded with sensors, "
        "software, and connectivity that enables them to collect and exchange data."
    ),
    "Cybersecurity": (
        "Cybersecurity is the practice of protecting systems, networks, and programs from digital "
        "attacks that aim to access, change, or destroy sensitive information."
    ),
}


def get_concept_names():
    return list(CONCEPTS.keys())


def get_reference_text(concept_name):
    return CONCEPTS.get(concept_name, "")
