"""
Custom Functions the Data Science Program
__author__ = James Irving, Brenda Hungerford
"""
__author__ = """James Irving"""
__email__ = 'james.irving.phd@gmail.com'
__version__ = '1.0.8'
from . import data_enrichment as data
from . import eda 
from . import evaluate
from . import insights 
from . import nlp

# from . import deployment_functions as deploy

from . import utils as utils


# from . import _eda_functions_plotly as eda_plotly

def show_code(function):
	"""Uses inspect modulem to retrieve source code for function.
	Displays as pthon-syntax Markdown code.
	
	Note: Python highlighting doesn't work correctly on VS Code or Google Colab
	"""
	import inspect
	from IPython.display import display, Markdown
	code = inspect.getsource(function)
	md = "```python" +'\n' + code + "\n" + '```' 
	display(Markdown(md))

