"""
Custom Functions the Data Science Program
__author__ = James Irving, Brenda Hungerford
"""
__author__ = """James Irving"""
__email__ = 'james.irving.phd@gmail.com'
__version__ = '1.1.2'
from . import data_enrichment as data
from . import eda 
from . import evaluate
from . import insights 
from . import nlp
from . import time_series

# from . import deployment_functions as deploy

from . import utils as utils


# from . import _eda_functions_plotly as eda_plotly

def show_code(function):
	"""
	Uses the inspect module to retrieve the source code for a function.
	Displays the code as Python-syntax Markdown code.

	Note: Python highlighting may not work correctly on some editors.

	Parameters:
	function (callable): The function for which to display the source code.

	Returns:
	None
	"""
	import inspect
	from IPython.display import display, Markdown

	code = inspect.getsource(function)
	md = "```python" +'\n' + code + "\n" + '```' 
	display(Markdown(md))

