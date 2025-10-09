"""
Sample artifact database for reference
Contains common archaeological finds organized by category
"""

def get_artifact_database():
    """
    Returns a dictionary of artifact categories with sample data
    This serves as a reference database for common archaeological finds
    """
    
    return {
        "Ancient Pottery": [
            {
                "name": "Ancient Greek Black-Figure Amphora",
                "value": "2500-15000",
                "age": "6th-5th Century BC",
                "description": "Large ceramic vessel used for storing wine or oil, characterized by black figures on red clay background"
            },
            {
                "name": "Roman Terra Sigillata Bowl",
                "value": "200-800",
                "age": "1st-3rd Century AD",
                "description": "Fine red pottery with glossy surface, often decorated with relief patterns"
            },
            {
                "name": "Medieval Cooking Pot",
                "value": "100-400",
                "age": "12th-15th Century AD",
                "description": "Utilitarian earthenware vessel used for cooking and food preparation"
            }
        ],
        
        "Ancient Coins": [
            {
                "name": "Roman Silver Denarius",
                "value": "50-500",
                "age": "1st Century BC - 3rd Century AD",
                "description": "Standard silver coin of the Roman Republic and early Empire"
            },
            {
                "name": "Ancient Greek Silver Tetradrachm",
                "value": "300-2000",
                "age": "5th-1st Century BC",
                "description": "Large silver coin worth four drachmae, often featuring gods or city symbols"
            },
            {
                "name": "Byzantine Gold Solidus",
                "value": "800-3000",
                "age": "4th-11th Century AD",
                "description": "Standard gold coin of the Byzantine Empire featuring imperial portraits"
            }
        ],
        
        "Stone Tools": [
            {
                "name": "Paleolithic Hand Axe",
                "value": "200-1500",
                "age": "300,000-30,000 years ago",
                "description": "Bifacially worked stone tool used for cutting and chopping"
            },
            {
                "name": "Neolithic Arrowhead",
                "value": "50-300",
                "age": "10,000-4,000 years ago",
                "description": "Finely crafted projectile point used for hunting"
            },
            {
                "name": "Stone Age Scraper",
                "value": "30-200",
                "age": "50,000-10,000 years ago",
                "description": "Stone tool used for processing hides and working wood"
            }
        ],
        
        "Metal Artifacts": [
            {
                "name": "Bronze Age Spearhead",
                "value": "300-1200",
                "age": "3000-1000 BC",
                "description": "Bronze weapon point with socketed hafting system"
            },
            {
                "name": "Iron Age Fibula",
                "value": "100-600",
                "age": "800-100 BC",
                "description": "Ancient brooch or safety pin used to fasten clothing"
            },
            {
                "name": "Medieval Pilgrim Badge",
                "value": "150-800",
                "age": "12th-16th Century AD",
                "description": "Lead-tin alloy badge worn by religious pilgrims"
            }
        ],
        
        "Decorative Objects": [
            {
                "name": "Roman Glass Unguentarium",
                "value": "200-800",
                "age": "1st-4th Century AD",
                "description": "Small glass vessel used for storing perfumes and oils"
            },
            {
                "name": "Ancient Egyptian Faience Amulet",
                "value": "300-2000",
                "age": "3000-300 BC",
                "description": "Glazed ceramic amulet often depicting gods or protective symbols"
            },
            {
                "name": "Viking Age Amber Bead",
                "value": "100-500",
                "age": "8th-11th Century AD",
                "description": "Fossilized resin bead used in jewelry and trade"
            }
        ]
    }

def search_artifact_by_name(name):
    """
    Search for an artifact by name in the database
    Returns matching artifacts across all categories
    """
    db = get_artifact_database()
    results = []
    
    for category, artifacts in db.items():
        for artifact in artifacts:
            if name.lower() in artifact['name'].lower():
                results.append({
                    'category': category,
                    **artifact
                })
    
    return results

def get_artifacts_by_category(category):
    """
    Get all artifacts from a specific category
    """
    db = get_artifact_database()
    return db.get(category, [])

def get_all_artifacts():
    """
    Get all artifacts from all categories as a flat list
    """
    db = get_artifact_database()
    all_artifacts = []
    
    for category, artifacts in db.items():
        for artifact in artifacts:
            all_artifacts.append({
                'category': category,
                **artifact
            })
    
    return all_artifacts
