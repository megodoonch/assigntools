import pickle
import bz2, gzip, lzma
import spacy
import requests, zipfile, io
from types import ModuleType

def read_pickle(path, method=None):
    """ reads variable pickled in the file and returns it
    """
    if method:
        m = eval(method)
        assert isinstance(m, ModuleType),\
            f"{m} should be a module"
        F = m.open(path, 'rb')
    else:
        F = open(path, 'rb')
    v = pickle.load(F)
    F.close()
    return v

def write_pickle(variable, path, method=None):
    """ pickle a given variable in the file
    """
    if method:
        m = eval(method)
        assert isinstance(m, ModuleType),\
            f"{m} should be a module"
        F = m.open(path, "wb")
    else:
        F = open(path, 'wb')  
    pickle.dump(variable, F)
    F.close()

def display_doc_dep(doc, d=150, compact=True):
    """ A shortcut function for displaying spaCy dependencies.
        It uses compact representation by default.
    """
    spacy.displacy.render(doc, style='dep', jupyter=True, \
                    options={'distance':d, 'fine_grained':True, 'compact':compact, 'collapse_punct':False})

def flatten_list(l):
    """ A recursive function that flatterns a list of lists with unbounded depth.
    """
    if not isinstance(l, list):
        return [l]
    return [ a for e in l for a in flatten_list(e) ]

def download_extract_zip(url):
    """ Download a file from url and extract its content in the current directory
        Return filenames of the extracted files  
    """
    r = requests.get(url) 
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall()
    return z.namelist()
