# -*- coding: utf-8 -*-
"""Convience module. 'from bs_ds.imports import *' will pre-load pd,np,plt,mpl,sns"""

from os import link


def global_imports(modulename,shortname = None, asfunction = False,check_vers=True):
        """from stackoverflow: https://stackoverflow.com/questions/11990556/how-to-make-global-imports-from-a-function,
        https://stackoverflow.com/a/46878490"""
        from importlib import import_module

        if shortname is None:
            shortname = modulename

        if asfunction is False:
            globals()[shortname] = import_module(modulename) #__import__(modulename)
        else:
            globals()[shortname] = eval(modulename + "." + shortname)
            
        if check_vers:
            return globals()[shortname].__version__

def clear():
    """Helper function to clear notebook display"""
    import IPython.display as dp
    return dp.clear_output()

def import_packages(import_list_of_tuples = None,  display_table=True, 
                    check_versions=True, link_text=None,
                    check_packages = ['matplotlib','seaborn','pandas','numpy','sklearn','dojo_ds'] ): #append_to_default_list=True, imports_have_description = True):
    """Uses the exec function to load in a list of tuples with:
    [('module','md','example generic tuple item')] formatting.
    >> Default imports_list:
    [('pandas',     'pd',   'High performance data structures and tools'),
    ('numpy',       'np',   'scientific computing with Python'),
    ('matplotlib',  'mpl',  "Matplotlib's base OOP module with formatting artists"),
    ('matplotlib.pyplot',   'plt',  "Matplotlib's matlab-like plotting module"),
    ('seaborn',     'sns',  "High-level data visualization library based on matplotlib"),
    ('IPython.display','dp','Display modules with helpful display and clearing commands.')
    ('dojo_ds','fs','Custom data science bootcamp student package')]
    """


    # import_list=[]
    from IPython.display import display
    import pandas as pd
    # if using default import list, create it:
    if (import_list_of_tuples is None): #or (append_to_default_list is True):
        import_list = [('Package','Handle','Documentation'),
                       ('pandas','pd', "https://pandas.pydata.org/docs/"),#'High performance data structures and tools'),
                       ('dojo_ds','ds',"https://fs-ds.readthedocs.io/en/latest/"),#'Custom data science bootcamp student package'), # 
                       ('numpy','np',"https://numpy.org/doc/stable/reference/"), # 'scientific computing with Python'), #
                       ('matplotlib','mpl','https://matplotlib.org/stable/api/index.html'),#"Matplotlib's base OOP module with formatting artists"), #
                       ('matplotlib.pyplot','plt',"https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.html#module-matplotlib.pyplot"),#"Matplotlib's matlab-like plotting module"), #
                       ('seaborn','sns',"https://seaborn.pydata.org/api.html"),#"High-level data visualization library based on matplotlib"), #
                    #    ('IPython.display','dp',"https://ipython.readthedocs.io/en/stable/api/generated/IPython.display.html")]#'Display modules with helpful display and clearing commands.')]#,
        ]
        # ('cufflinks','cf','Adds df.iplot() interactive Plotly figs. To use, run >> cf.go_offline()')]

    # if using own list, rename to 'import_list'
    else:
        import_list = import_list_of_tuples


    # Use exec command to create global handle variables and then load in package as that handle
    for package,handle,_ in import_list[1:]:
        # old way: # exec(f'import {package} as {handle}')
        # global_imports(package,handle)
         
        global_imports(package,handle,check_vers=False)


    ## Make dataframe of imports
    # create and return styled dataframe
    # columns=['Package','Handle','Documentation']
    df_imports= pd.DataFrame(import_list[1:], columns=import_list[0])
    
    ## Reorder Columns
    # df_imports = df_imports[['Handle','Package','Documentation']]
    
    
  
    
    
    ## make dataframe of versions
    if check_versions:
        pkg_vers_df = check_package_versions(packages=check_packages,
                                             show_only=False,fpath=False)
        
        ## Adding Imported column to track non-imported versions
        df_imports['Imported'] = 'Y'
        df_imports = pd.merge(df_imports, pkg_vers_df,on='Package',how='outer')
        
        df_imports = df_imports[['Package','Handle','Version','Documentation','Imported',]]
        # df_imports.insert(1,'',' as ')
        # df_imports = df_imports.rename({'Package':'import','Handle':'as'},axis=1)
        df_imports['Imported'] = df_imports['Imported'].fillna('N')
        df_imports = df_imports.fillna('')
    

        # display(pkg_vers_df.style.hide_index().set_caption('Package Version Report')) 

    # Display summary dataframe
    if display_table==True:            
        ## Create Columns Names List

        # create and return styled dataframe
        # columns=['Package','Handle','Documentation']
        # df_imported= pd.DataFrame(import_list, columns=columns)
        # # df_imported=pd.concat([df_imported['Handle'],df_imported[['Package','Documentation']]],axis=1)
        
        # # df_imports = pd.merge(df_imported,pkg_vers_df,on='Package')
        # # df_imports = df_imports[['Handle','Package','Documentation','Version']]
        # df_imports = df_imported[['Handle','Package','Documentation']]
        #.sort_values('Package').
        import dojo_ds as ds
        try:
            print(f"dojo_ds v{ds.__version__} loaded.")#  Read the docs: https://fs-ds.readthedocs.io/en/latest/ ")
        except:
            pass
        try:
            dfs = df_imports.style.hide_index().set_caption('Loaded Packages & Info')
        except:
            dfs = df_imports.style.hide().set_caption('Loaded Packages & Info')
        
        ## Determine if links will have display text
        if link_text is None:
            kwargs = {'Documentation':clickable}

        else:
            kwargs = {'Documentation':lambda x:  clickable(x,link_text)}

        ## apply kwargs above and set additional properties
        display(dfs.format(kwargs).set_properties(**{'text-align':'left'}).\
                                set_properties(subset=['Imported','Handle'],**{'text-align':'center'}) )
            # display(dfs.format({'Documentation':lambda x:  clickable(x,link_text)}))
        # return df_imports

    # or just print statement
    else:
        print('Modules successfully loaded.')
        

def clickable(path,label=None):
    """Adapted from: https://www.geeksforgeeks.org/how-to-create-a-table-with-clickable-hyperlink-to-a-local-file-in-pandas/"""
    # returns the final component of a url
    # f_url = os.path.basename(path)
    if label is None:
    # convert the url into link
        return '<a href="{}">{}</a>'.format(path, path)
    else: 
        return '<a href="{}">{}</a>'.format(path, label)  



def check_package_versions(packages = ['matplotlib','seaborn','pandas','numpy','sklearn','dojo_ds'],
                           fpath=False, show_only=True):
    """Imports packages and saves the name and version number to a dataframe"""
    import pandas as pd
    import inspect
    version_list = [['Package','Version']]
    
    ## Remove submodules from version check (wont have version #)
    for package in packages:
        if '.' not in package:
            try:
                ## use global imports and retrieve version #
                vers = global_imports(package,None,check_vers=True)
            except:
                vers = '!'
            version_list.append([package,vers])

    # Convert to df
    pkg_vers_df = pd.DataFrame(version_list[1:],columns=version_list[0])
    
    ## If get_fpath
    if fpath==True:
        pkg_vers_df['File'] = pkg_vers_df['Package'].map(lambda x: inspect.getsourcefile(globals()[x]))
        # for package in packages:
    
    if show_only==True: 
        if fpath==True:   
            dfs = pkg_vers_df.style.set_properties(subset='File',
                                                **{'width':"600px","text-align":'center'})
        else:
            dfs = pkg_vers_df.style
        display(dfs.set_caption('Package Versions'))
    
        
    else:
        return pkg_vers_df



try:
    from IPython.display import clear_output
    clear_output()
except:
    pass
finally:
    # fs = None
    import_packages()
    
# try:
#     import cufflinks as cf 
#     cf.go_offline()
#     print('[i] Pandas .iplot() method activated.')
# except:
#     pass

    