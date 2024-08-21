"""
Custom Functions the Data Science Program
__author__ = James Irving, Brenda Hungerford
"""
__author__ = """James Irving"""
__email__ = 'james.irving.phd@gmail.com'
__version__ = '1.1.12'
from . import data_enrichment as data
from . import eda as eda
from . import evaluate as evaluate
from . import insights  as insights
from . import nlp as nlp
from . import time_series as time_series
from . import datasets as datasets
from . import fileinfo as fileinfo

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

