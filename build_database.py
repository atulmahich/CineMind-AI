import os
import pickle
import numpy as np
import pandas as pd
import faiss

from sentence_transformers import (
    SentenceTransformer
)

# =====================================================
# CREATE FOLDERS
# =====================================================

os.makedirs(
    "vector_store",
    exist_ok=True
)

# =====================================================
# TV KNOWLEDGE DATABASE
# =====================================================

documents = [

# =====================================================
# THE BLACKLIST
# =====================================================

{
"series":"The Blacklist",
"season":1,
"episode":1,
"title":"Pilot",
"context":
"""
Reddington surrenders to the FBI
and demands to work only with
Elizabeth Keen.
"""
},

{
"series":"The Blacklist",
"season":1,
"episode":1,
"title":"Pilot",
"context":
"""
Reddington introduces
the blacklist,
a secret list of dangerous
criminals.
"""
},

{
"series":"The Blacklist",
"season":1,
"episode":1,
"title":"Pilot",
"context":
"""
Ranko Zamani is identified
as a dangerous criminal
targeting a general's daughter.
"""
},

# =====================================================
# BREAKING BAD
# =====================================================

{
"series":"Breaking Bad",
"season":1,
"episode":1,
"title":"Pilot",
"context":
"""
Walter White discovers he has
lung cancer and starts making
methamphetamine to support
his family.
"""
},

{
"series":"Breaking Bad",
"season":2,
"episode":8,
"title":"Better Call Saul",
"context":
"""
Saul Goodman becomes Walter
White's lawyer and helps him
expand his illegal drug business.
"""
},

{
"series":"Breaking Bad",
"season":5,
"episode":16,
"title":"Felina",
"context":
"""
Walter White sacrifices himself
after rescuing Jesse Pinkman.
"""
},

# =====================================================
# THE BOYS
# =====================================================

{
"series":"The Boys",
"season":1,
"episode":1,
"title":"The Name of the Game",
"context":
"""
Starlight joins The Seven while
Homelander is introduced as
its dangerous leader.
"""
},

{
"series":"The Boys",
"season":2,
"episode":8,
"title":"What I Know",
"context":
"""
Starlight helps expose
Homelander and Vought.
"""
},

{
"series":"The Boys",
"season":3,
"episode":8,
"title":"The Instant White-Hot Wild",
"context":
"""
Butcher and Homelander
continue their rivalry.
"""
},

# =====================================================
# GAME OF THRONES
# =====================================================

{
"series":"Game of Thrones",
"season":1,
"episode":9,
"title":"Baelor",
"context":
"""
Ned Stark is publicly executed
on King Joffrey's orders.
"""
},

{
"series":"Game of Thrones",
"season":3,
"episode":9,
"title":"The Rains of Castamere",
"context":
"""
The Red Wedding results in
the deaths of Robb Stark,
Talisa and Catelyn Stark.
"""
},

{
"series":"Game of Thrones",
"season":8,
"episode":6,
"title":"The Iron Throne",
"context":
"""
Jon Snow kills Daenerys
to stop her violent rule.
"""
},

# =====================================================
# FRIENDS
# =====================================================

{
"series":"Friends",
"season":1,
"episode":1,
"title":"Pilot",
"context":
"""
Rachel leaves Barry at the altar
and reconnects with Monica.
"""
},

{
"series":"Friends",
"season":5,
"episode":14,
"title":"Everybody Finds Out",
"context":
"""
The group discovers Monica and
Chandler are secretly dating.
"""
},

{
"series":"Friends",
"season":10,
"episode":18,
"title":"The Last One",
"context":
"""
Ross and Rachel reunite.
The friends leave the apartment.
"""
},

# =====================================================
# STRANGER THINGS
# =====================================================

{
"series":"Stranger Things",
"season":1,
"episode":1,
"title":"Chapter One",
"context":
"""
Will Byers disappears and Eleven
appears mysteriously.
"""
},

{
"series":"Stranger Things",
"season":4,
"episode":9,
"title":"The Piggyback",
"context":
"""
Eleven battles Vecna while
Max is critically injured.
"""
}
]

# =====================================================
# DATAFRAME
# =====================================================

df = pd.DataFrame(
    documents
)

print(
    f"Loaded {len(df)} entries"
)

# =====================================================
# EMBEDDINGS
# =====================================================

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

texts = df["context"].tolist()

embeddings = model.encode(
    texts
)

embeddings = np.array(
    embeddings
).astype("float32")

# =====================================================
# FAISS
# =====================================================

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(
    dimension
)

index.add(
    embeddings
)

# =====================================================
# SAVE
# =====================================================

faiss.write_index(
    index,
    "vector_store/faiss_index.bin"
)

with open(
    "vector_store/metadata.pkl",
    "wb"
) as f:

    pickle.dump(
        documents,
        f
    )

print(
    "✅ Entertainment database built"
)
