import json
import shutil
import streamlit as st
import pandas as pd
import pathlib
from scipy.cluster.hierarchy import linkage, leaves_list
import plotly.graph_objects as go
import stages.utils.pdaa as pdaa
import stages.utils.pubchem as pubchem
import stages.utils.sparql as sparql
import stages.utils.simple_cache as simple_cache
import rdflib
from concurrent.futures import ThreadPoolExecutor

# Setup
cachedir = pathlib.Path("cache") / "notebooks" / "pdaa"
cachedir.mkdir(parents=True, exist_ok=True)
pubchemtools = pubchem.PubchemTools()

# Utility Functions
@simple_cache.simple_cache_df(cachedir / "get_uri_titles")
def get_uri_titles():
    uri_titles = (
        sparql.Query(pdaa.pdaa_graph, cachedir / "pdaa_graph")
        .select_typed({"uri": str, "title": str})
        .where(f"?ppuri <http://purl.org/dc/elements/1.1/title> ?title")
        .where(f"?ppuri <{rdflib.RDF.type}> toxindex:predicted_property")
        .where(f"?ppuri <http://purl.org/dc/elements/1.1/has_identifier> ?uri")
        .cache_execute()
        .groupby("uri")
        .first()
        .reset_index()
    )
    # Ensure unique titles
    for title in uri_titles["title"].value_counts()[uri_titles["title"].value_counts() > 1].index:
        mask = uri_titles["title"] == title
        uri_titles.loc[mask, "title"] = [f"{title}_{i+1}" for i in range(sum(mask))]
    return uri_titles
uri_titles = get_uri_titles()

all_ao = sparql.Query(pdaa.pdaa_graph, cachedir / 'pdaa_graph') \
    .select_typed({'aop':str, 'mie':str, 'ao':str, 'ao_title':str, 'mie_title':str}) \
    .where(f'?aop <{rdflib.RDF.type}> aop:AdverseOutcomePathway') \
    .where(f'?aop aop:has_adverse_outcome ?ao') \
    .where(f'?aop aop:has_molecular_initiating_event ?mie') \
    .where(f'?ao <http://purl.org/dc/elements/1.1/title> ?ao_title') \
    .where(f'?mie <http://purl.org/dc/elements/1.1/title> ?mie_title') \
    .cache_execute()

@simple_cache.simple_cache_df(cachedir / "link_predicted_properties_to_mies")
def link_predicted_properties_to_mies():
    predicted_property_identifiers = pdaa.proptoken_uris['uri'].unique()
    mies = all_ao['mie'].unique()
    mie_proptoken_id_simtable = pdaa.get_uri_similars(mies, predicted_property_identifiers)
    mie_proptoken_id_simtable.columns = ['mie', 'property_token_id_uri', 'similarity']
    mie_proptoken_id_simtable['mie'].nunique()
    mie_proptoken_id_simtable['property_token_id_uri'].nunique()
    
    # we want to keep the top property for each mie
    resdf = (mie_proptoken_id_simtable
             .sort_values('similarity', ascending=False)
             .groupby('mie')
             .head(3)
             .reset_index(drop=True))
    resdf['property_token_id_uri'].nunique()
    print(pd.qcut(resdf['similarity'], q=10).value_counts().sort_index())
    return resdf

mie_proptoken_id_simtable = link_predicted_properties_to_mies()

def parse_chemical_list(chemical_list) -> dict[str, str]:
    chemicals = {}
    for line in filter(None, map(str.strip, chemical_list.split("\n"))):
        alias, name = line.split(":", 1) if ":" in line else (line, line)
        chemicals[alias.strip()] = name.strip()
    return chemicals

def build_heatmap(predictions, midvalue=0.5, max_val=None):
       
    # Pivot the predictions into a matrix
    pdf = predictions.pivot(index='title', columns='chemical_name', values='prediction')

    # Perform hierarchical clustering on rows and columns
    row_linkage = linkage(pdf.values, method='ward', metric='euclidean')
    col_linkage = linkage(pdf.T.values, method='ward', metric='euclidean')

    # Get the order of rows and columns based on clustering
    row_order = leaves_list(row_linkage)
    col_order = leaves_list(col_linkage)

    # Reorder the DataFrame
    pdf_clustered = pdf.iloc[row_order, col_order]

    # Create clickable row labels using the 'uri' column
    row_links = dict(zip(predictions['title'], predictions['uri']))
    row_labels = [f'<a href="{row_links[title]}" target="_blank">{title}</a>' if title in row_links else title for title in pdf_clustered.index]

    # Get min and max values for colorscale
    min_val = pdf_clustered.values.min()
    max_val = pdf_clustered.values.max() if max_val is None else max_val

    # rescale z so that if it is bigger than max_val, it is set to max_val and if it is smaller than min_val, it is set to min_val
    pdf_clustered_clipped = pdf_clustered.clip(min_val, max_val)

    # Create blue gradient up to midvalue
    colorscale = [[0, 'rgb(0,0,255)'],[1, 'rgb(255,0,0)']]
    # Define the heatmap trace with dynamic min/max values
    heatmap = go.Heatmap(
        z=pdf_clustered_clipped.values,
        x=pdf_clustered_clipped.columns,
        y=row_labels,
        colorscale=colorscale,  # Blue to black to red
        zmin=min_val,
        zmax=max_val,
        colorbar=dict(title="", orientation="v", x=-0.2, y=0.5, thickness=20),
        xgap=2,  # Increased gap between cells to make boxes appear smaller
        ygap=2,  # Increased gap between cells to make boxes appear smaller
        showscale=True,
        text=[[f'{val:.2f}' for val in row] for row in pdf_clustered.values],  # Show values in cells
        texttemplate='%{text}',
        textfont={"color": "white"}  # White text
    )

    # Create the layout
    layout = go.Layout(
        title=f'Property Predictions Heatmap',
        xaxis=dict(
            title="Chemical Names",
            gridcolor='black',  # Make x-axis grid black
            showgrid=False
        ),
        yaxis=dict(
            title="Titles", 
            tickmode="array", 
            tickvals=list(range(len(pdf.index))), 
            ticktext=row_labels,
            side='right',  # Move labels to right side
            gridcolor='black',  # Make y-axis grid black
            showgrid=False,
            tickfont=dict(size=16),  # Adjust font size if needed
            tickprefix='   ',  # Add padding after labels
            ticksuffix='     '  # Add padding before labels
        ),
        height=800,
        plot_bgcolor='black',  # Make gaps appear black
        margin=dict(r=350)  # Increased right margin for more space
    )

    # Generate the figure
    return go.Figure(data=[heatmap], layout=layout)

def build_barchart(predictions, title):
    fig = go.Figure()
    
    # Create the bar chart
    data = predictions.groupby('chemical_name')['prediction'].sum().sort_values(ascending=False)
    
    fig.add_trace(go.Bar(
        x=data.index,
        y=data.values,
        marker=dict(
            color=data.values,
            colorscale=[[0, 'rgb(0,0,255)'], [1, 'rgb(255,0,0)']],
            line=dict(color='white', width=1)
        )
    ))
    
    # Update the layout
    fig.update_layout(
        title=title,
        xaxis_title='Chemical',
        yaxis_title='Sum of Predictions',
        xaxis_tickangle=45,
        yaxis_gridcolor='white',
        plot_bgcolor='black',
        paper_bgcolor='black',
        font=dict(color='white'),
        height=400,
        xaxis=dict(
            gridcolor='white',
            showgrid=True
        ),
        yaxis=dict(
            gridcolor='white', 
            showgrid=True
        )
    )
    
    return fig

# shutil.rmtree(cachedir / "get_all_predictions")
@simple_cache.simple_cache_df(cachedir / "get_all_predictions")
def get_all_predictions(chemical_list):

    chemicals = parse_chemical_list(chemical_list)
    inchi_list = []
    for chem in chemicals.values():
        try:
            inchi = pubchemtools.lookup_chemical_inchi(chem)
            inchi_list.append(inchi)
        except Exception as e:
            raise Exception(f"InChI not found for {chem}. Please pick another name from https://pubchem.ncbi.nlm.nih.gov/")
    
    inchi2name = dict(zip(inchi_list, chemicals.keys()))

    all_predictions = []
    with ThreadPoolExecutor() as executor:
        for inchi in inchi_list:
            all_predictions.extend(executor.submit(pdaa.predict_all_properties_with_sqlite_cache, [inchi]).result())

    preds_df = pd.DataFrame(all_predictions, columns=['inchi', 'token', 'prediction'])
    preds_df = preds_df.merge(pdaa.proptoken_uris[['uri','token']], on='token')[['uri','inchi','prediction']]
    preds_df.rename(columns={'uri':'property_token_id_uri'}, inplace=True)
    preds_df["chemical_name"] = preds_df["inchi"].map(inchi2name)
    
    all_pred_df = preds_df.merge(mie_proptoken_id_simtable, on='property_token_id_uri')
    all_pred_df['weight'] = all_pred_df['similarity'] * all_pred_df['prediction']

    return all_pred_df[['mie','property_token_id_uri','chemical_name','similarity','prediction','weight']]

@simple_cache.simple_cache_df(cachedir / "get_property_predictions")
def get_property_predictions(predictions, prompt, top_n=20):
    # get relevant property predictions
    prompt_property = pdaa.get_prompt_similars(prompt, pdaa.proptoken_uris['uri'].unique(), top_k_to_search=10000)
    prompt_property = prompt_property.query('similarity > 0.2').sort_values('similarity', ascending=False).head(top_n)
    
    res = predictions[['property_token_id_uri','chemical_name','prediction']].drop_duplicates()
    res = res[res['property_token_id_uri'].isin(prompt_property['uri'])]
    res = res.merge(uri_titles, left_on='property_token_id_uri', right_on='uri')
    res = res[['uri','title','chemical_name','prediction']]

    # find the duplicated chemical_name, title pairs
    # dup_mask = res.duplicated(subset=['chemical_name','title'], keep=False)
    # dup_res = res[dup_mask]
    # dup_res['title'].value_counts()

    return res[['uri','title','chemical_name','prediction']]

@simple_cache.simple_cache_df(cachedir / "get_mie_predictions")
def get_mie_predictions(predictions, prompt, top_n=20):

    # get relevant aop, mie that lead to these should be included    
    prompt_aop = pdaa.get_prompt_similars(prompt, all_ao['aop'].unique(), top_k_to_search=10000)
    prompt_aop = prompt_aop.query('similarity > 0.2')
    prompt_aop = all_ao.merge(prompt_aop, left_on='aop', right_on='uri')[['mie','mie_title','similarity']]
    
    # get relevant adverse outcomes, mie that lead to these should be included    
    prompt_ao = pdaa.get_prompt_similars(prompt, all_ao['ao'].unique(), top_k_to_search=10000)
    prompt_ao = prompt_ao.query('similarity > 0.2')
    prompt_ao = all_ao.merge(prompt_ao, left_on='ao', right_on='uri')[['mie','mie_title','similarity']]

    # get relevant mie predictions
    prompt_mie = pdaa.get_prompt_similars(prompt, all_ao['mie'].unique(), top_k_to_search=10000)
    prompt_mie = prompt_mie.query('similarity > 0.2')
    prompt_mie = all_ao.merge(prompt_mie, left_on='mie', right_on='uri')[['mie','mie_title','similarity']]
    prompt_mie = prompt_mie.merge(prompt_ao, on='mie').merge(prompt_aop, on='mie')
    prompt_mie = prompt_mie.groupby(['mie','mie_title']).agg({'similarity':'max'}).reset_index()

    # get the top n mie in the predictions df
    pred_mie = predictions['mie'].unique()
    prompt_mie = prompt_mie[prompt_mie['mie'].isin(pred_mie)].sort_values('similarity', ascending=False).head(top_n)
    prompt_mie = prompt_mie[['mie','mie_title']]

    mie_predictions = predictions[['mie','property_token_id_uri','chemical_name','prediction']].merge(prompt_mie, on=['mie'])
    mie_predictions = mie_predictions[['mie','mie_title','property_token_id_uri','chemical_name','prediction']]

    # finally get the mean prediction across all linked property_token_id_uri
    mie_df = mie_predictions.groupby(['mie','mie_title','chemical_name'])['prediction'].mean().reset_index()
    
    # need 'uri','title','chemical_name','prediction'
    mie_weights = mie_df[['mie','mie_title','chemical_name','prediction']]
    mie_weights.rename(columns={'mie':'uri','mie_title':'title'}, inplace=True)

    # make unique short titles
    title_df = mie_weights[['title']].drop_duplicates()
    title_df['short_title'] = title_df['title'].str[:40]
    dup_mask = title_df.duplicated(subset=['short_title'], keep=False)
    title_df.loc[dup_mask, 'short_title'] += ' ' + title_df[dup_mask].groupby('short_title').cumcount().add(1).astype(str)
    
    mie_weights_title = mie_weights.merge(title_df, on='title')
    mie_weights_title['title'] = mie_weights_title['short_title']
    mie_weights_title.drop('short_title', axis=1, inplace=True)

    return mie_weights_title[['uri','title','chemical_name','prediction']]

# UI Setup
st.set_page_config(page_title="Heatmap App", layout="wide")


st.title("Find & Rank Adverse Outcomes for a Chemical & Prompt")

examples = json.load(open('streamlit/examples.json'))['examples']

# Initialize session state variables if they don't exist
if 'chemical_list' not in st.session_state:
    st.session_state.chemical_list = examples[5]['chemicals']
if 'prompt' not in st.session_state:
    st.session_state.prompt = examples[5]['prompt']

# Use session state values to initialize widgets
chemical_list = st.text_area("Enter `alias:name` pairs", st.session_state.chemical_list, height=200)
prompt = st.text_input("Enter a prompt (e.g., 'endocrine disruption')", st.session_state.prompt)

if 'updating_charts' not in st.session_state:
    st.session_state.updating_charts = False

cols = st.columns(3)
for i, example in enumerate(examples):
    if i % 3 == 0 and i > 0:
        cols = st.columns(3)
    if cols[i % 3].button(example['title'], key=f"example_button_{i}", disabled=st.session_state.updating_charts):
        st.session_state.prompt = example['prompt']
        st.session_state.chemical_list = example['chemicals']
        

# Style "generate heatmap" button to be in the center, with background color"
st.markdown(
    """
    <style>
    [data-testid="stButton"] button {
        display: block;
        justify: center;
        margin-left: auto;
        margin-right: auto;
        align-items: center;
        width: 100%;

        # background-color: #f0f2f6 !important;
        # border-color: #ff0000 !important;      
    }
        
    </style>
    """,
    unsafe_allow_html=True,
)

if st.button("Generate Heatmaps", disabled=st.session_state.updating_charts):
    
    st.session_state.updating_charts = True
    
    # Style tab headings to be larger
    st.markdown(
        """
        <style>
        /* Set a grey background for the tab container */
        [data-baseweb="tab-list"] {
            background-color: #f0f2f6 !important;
            border-radius: 10px;
            padding: 8px;
        }
        
        /* Set a grey background for individual tabs */
        [data-baseweb="tab"] {
            background-color: #f0f2f6 !important;
            border-radius: 10px;
            margin: 10px;
        }
        
        /* Increase the font size for tab headers */
        [data-baseweb="tab"] p {
            font-size: 15pt !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Property Heatmap", "MIE Heatmap", "Aggregate Predictions"])

    # Tab 1 : Property Heatmap
    with tab1: 
        try:
            predictions = get_all_predictions(chemical_list)
            property_predictions = get_property_predictions(predictions, prompt)

            # Build plot
            st.subheader("Property Heatmap")
            fig_property = build_heatmap(property_predictions)
            st.plotly_chart(fig_property, use_container_width=True)

        except Exception as e:
            st.error("Unable to generate heatmaps. Please check that your prompt and chemicals are valid.")
            st.error(e)

    # Tab 2 : MIE Heatmap
    with tab2: 
        try: 
            mie_predictions = get_mie_predictions(predictions, prompt)
            st.subheader("MIE Heatmap") 
            # print(f'there are {len(mie_predictions)} mie predictions')
            # st.write(f'There are {len(mie_predictions)} mie predictions')

            fig_mie = build_heatmap(mie_predictions)
            st.plotly_chart(fig_mie, use_container_width=True)
        
        except Exception as e:
            st.error("Unable to generate heatmaps. Please check that your prompt and chemicals are valid.")
            st.error(e)

    # Tab 3 : Aggregate Predictions
    with tab3:
        try:
            st.subheader("Aggregate Predictions")
            aggregate_predictions = mie_predictions.groupby(['chemical_name'])['prediction'].sum().reset_index()
            fig_aggregate = build_barchart(
                aggregate_predictions, title="Aggregate MIE Predictions")
            st.plotly_chart(fig_aggregate, use_container_width=True)

        except Exception as e:
            st.error("Unable to generate heatmaps. Please check that your prompt and chemicals are valid.")
            st.error(e)
    
    st.session_state.updating_charts = False