import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from itertools import combinations
import plotly.express as px
from sklearn.model_selection import train_test_split
from plotly.subplots import make_subplots
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import scipy.cluster.hierarchy as sch
st.title("BAJA PROJEKT")
st.set_option('deprecation.showPyplotGlobalUse', False)
def confidence_ellipseee(x, y, n_std=1.96, size=100):
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    theta = np.linspace(0, 2 * np.pi, size)
    ellipse_coords = np.column_stack([ell_radius_x * np.cos(theta), ell_radius_y * np.sin(theta)])

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    x_scale = np.sqrt(cov[0, 0]) * n_std
    x_mean = np.mean(x)

    # calculating the stdandard deviation of y ...
    y_scale = np.sqrt(cov[1, 1]) * n_std
    y_mean = np.mean(y)

    translation_matrix = np.tile([x_mean, y_mean], (ellipse_coords.shape[0], 1))
    rotation_matrix = np.array([[np.cos(np.pi / 4), np.sin(np.pi / 4)],
                                [-np.sin(np.pi / 4), np.cos(np.pi / 4)]])
    scale_matrix = np.array([[x_scale, 0],
                             [0, y_scale]])
    ellipse_coords = ellipse_coords.dot(rotation_matrix).dot(scale_matrix) + translation_matrix

    path = f'M {ellipse_coords[0, 0]}, {ellipse_coords[0, 1]}'
    for k in range(1, len(ellipse_coords)):
        path += f'L{ellipse_coords[k, 0]}, {ellipse_coords[k, 1]}'
    path += ' Z'
    return path

uploaded_file = st.file_uploader("If Dataset has an y column it has to be last, the dataset should also contain an id column and it should be first",type=['csv'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file,delimiter=',')

    st.dataframe(df)
    option0=st.sidebar.selectbox('What kind of plots do you want to have',('PCA','HCA','HCA 2D'))
    if option0=='PCA':
        option1=st.sidebar.selectbox('Does your dataset contain Y column',('','Yes','No'))
        if option1=='Yes':
            option2=st.sidebar.selectbox('What kind of Y column does your data set have',('','Categorical','Numerical'))
            option3=st.sidebar.selectbox("How many components do you want to use?",list(np.arange(len(df.columns)-2,1,-1)))
            template=st.sidebar.selectbox("Select plot style",("plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn", "simple_white", "none"))
            number_of_X_columns=len(df.columns)-1
            skladowe=option3
            comb = list(combinations(list(range(0, skladowe)), 2))
            plot_font_size=st.sidebar.selectbox("Select font size for plots",list(np.arange(10,31,1)))
            Gen=st.sidebar.button("Generate")
            if Gen:
                if option2!='':
                    X = df.iloc[:, 1:number_of_X_columns].values
                    Y = df.iloc[:, [number_of_X_columns]].values
                    Y_un = np.unique(Y)

                    # We standarize the X columns
                    Z = X
                    X = StandardScaler().fit_transform(X)
                    pcaModel = PCA(n_components=skladowe)
                    pca = pcaModel.fit(X)
                    principalComponents = pca.transform(X)
                    st.write("Percentege of variance explained by every PCA: ")
                    Pca_var=pd.DataFrame(pca.explained_variance_ratio_.cumsum())
                    Pca_var.columns=["Percentege of variance explained by every PCA: "]
                    Pca_var['PCA']=list(np.arange(1,skladowe+1))
                    st.dataframe(Pca_var)



                    fig = px.line(x=list(range(0, skladowe)), y=pca.explained_variance_ratio_, markers=True, labels={
                        "x": "Principal Component",
                        "y": "Variance Explained",

                    })

                    fig.update_layout(title_text='Scree Plot', title_x=0.5,template=template, font=dict(size=plot_font_size))
                    st.plotly_chart(fig,use_container_width=True)

                    X_un = df.columns[1:len(df.columns) - 1]

                    principalDf = pd.DataFrame(data=principalComponents
                                               )
                    finalDf = principalDf
                    newColumns=[]
                    for i in range(len(finalDf.columns)):
                        newColumns.append("PC "+str(i+1))
                    finalDf.columns=newColumns
                    finalDf['target'] = df.iloc[:, [number_of_X_columns]].values


                    for i in range(len(finalDf.columns) - 1):
                        s = principalComponents[:, i]

                        scalex = (1.0 / (s.max() - s.min()))

                        sx = s * scalex

                        finalDf['scaled_PC_' + str(i + 1)] = sx.tolist()

                    finalDf['indeks'] = finalDf.index
                    new_number_of_X_columns = len(finalDf.columns)

                    fig = go.Figure()

                    fig.add_trace(

                        go.Scatter(
                            x=finalDf.iloc[:, skladowe + comb[0][0] + 1],
                            y=finalDf.iloc[:, skladowe + comb[0][1] + 1],
                            text=finalDf.indeks.tolist(),
                            hovertemplate=
                            "Index: %{text}" +
                            "<br>" + str(finalDf.columns[skladowe + comb[0][0] + 1]) + ":  " + "%{x}" +
                            "<br>" + str(finalDf.columns[skladowe + comb[0][1] + 1]) + ":  " + "%{y}"

                            ,
                            mode='markers',
                            showlegend=False

                        )
                    )
                    for i in range(len(X_un)):
                        fig.add_trace(

                            go.Scatter(
                                x=[0, np.transpose(pcaModel.components_)[i][0]],
                                y=[0, np.transpose(pcaModel.components_)[i][1]],
                                text=X_un.tolist(),
                                hovertemplate="%{text}<br>" +
                                              str(finalDf.columns[skladowe + comb[0][0]]) + ":  %{x}<br>" +
                                              str(finalDf.columns[skladowe + comb[0][1]]) + ":  %{y}",
                                mode="lines+markers",
                                showlegend=False
                            )

                        )

                    fig.update_layout(
                        title={
                            'text': "Factor Loadings Graph",
                            'y': 0.9,
                            'x': 0.5,
                            'xanchor': 'center',
                            'yanchor': 'top'},
                        template=template,
                        xaxis={
                            'title': str(finalDf.columns[skladowe + comb[0][0] + 1])
                        },
                        yaxis={'title': str(finalDf.columns[skladowe + comb[0][1] + 1])},
                        font=dict(size=plot_font_size))

                    st.plotly_chart(fig, use_container_width=True)

                    CorData = pd.DataFrame(np.transpose(pcaModel.components_))
                    CorData.index = X_un
                    NewColumns = []

                    for i in range(len(CorData.columns)):
                        NewColumns.append("PC_" + str(i + 1))
                    CorData.columns = NewColumns

                    fig=px.imshow(CorData,labels=dict(color="Corelation"))

                    fig.update_layout( font=dict(size=plot_font_size))

                    st.plotly_chart(fig, use_container_width=True)

                    if option2 == "Categorical":
                        for j in range(len(comb)):
                            fig = go.Figure()
                            for i in range(len(np.unique(finalDf["target"]))):
                                viewfinalDf = finalDf[finalDf.target == np.unique(finalDf.target)[i]]
                                fig.add_trace(

                                    go.Scatter(
                                        x=viewfinalDf.iloc[:, skladowe + comb[j][0] + 1],
                                        y=viewfinalDf.iloc[:, skladowe + comb[j][1] + 1],
                                        text=viewfinalDf.indeks.tolist(),
                                        hovertemplate=str(np.unique(finalDf.target).tolist()[i]) + "<br>" +
                                                      "Index: %{text}" +
                                                      "<br>" + str(finalDf.columns[skladowe + comb[j][0] + 1]) + ": " + "%{x}" +
                                                      "<br>" + str(finalDf.columns[skladowe + comb[j][1] + 1]) + ": " + "%{y}"

                                        ,
                                        mode='markers',
                                        showlegend=False

                                    )
                                )

                                fig.add_shape(type='path',
                                              path=confidence_ellipseee(viewfinalDf.iloc[:, skladowe + comb[j][0] + 1],
                                                                        viewfinalDf.iloc[:, skladowe + comb[j][1] + 1]),
                                              line={'dash': 'dot'},
                                              )

                                fig.update_layout(
                                    xaxis={
                                        'title': str(finalDf.columns[skladowe + comb[j][0] + 1])
                                    },
                                    template=template,
                                    yaxis={'title': str(finalDf.columns[skladowe + comb[j][1] + 1])},
                                    font=dict(size=plot_font_size))

                            st.plotly_chart(fig, use_container_width=True)
                        st.dataframe(finalDf)


                        def convert_df(df):
                            return df.to_csv(index=False).encode('utf-8')


                        csv = convert_df(finalDf)
                        st.download_button(
                            "Press to Download",
                            csv,
                            "file.csv",
                            "text/csv",
                            key='download-csv'
                        )
                        number_of_X_columns = len(df.columns) - 1
                        X = df.iloc[:, 0:number_of_X_columns].values
                        Y = df.iloc[:, [number_of_X_columns]].values

                        Xtrain, Xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.2)

                        sc = StandardScaler()
                        Xtrain = sc.fit_transform(Xtrain)
                        Xtest = sc.transform(Xtest)

                        pca = PCA(n_components=skladowe)
                        Xtrain = pca.fit_transform(Xtrain)
                        Xtest = pca.transform(Xtest)

                        from sklearn.linear_model import LogisticRegression

                        classifier = LogisticRegression(random_state=0)
                        classifier.fit(Xtrain, ytrain)
                        ypred = classifier.predict(Xtest)

                        from sklearn.metrics import confusion_matrix

                        cm = confusion_matrix(ytest, ypred)
                        cm_df = pd.DataFrame(cm,
                                             index=Y_un,
                                             columns=Y_un)

                        fig = px.imshow(cm_df,labels=dict(color="Points"))
                        fig.update_layout(font=dict(size=plot_font_size))
                        st.plotly_chart(fig, use_container_width=True)





                    if option2 == "Numerical":
                        for j in range(len(comb)):
                            fig = go.Figure()

                            fig.add_trace(

                                go.Scatter(
                                    x=finalDf.iloc[:, skladowe + comb[j][0] + 1],
                                    y=finalDf.iloc[:, skladowe + comb[j][1] + 1],

                                    text=finalDf.indeks.tolist(),
                                    hovertemplate=
                                    "Index: %{text}" +
                                    "<br>" + str(finalDf.columns[skladowe + comb[j][0] + 1]) + ": " + "%{x}" +
                                    "<br>" + str(finalDf.columns[skladowe + comb[j][1] + 1]) + ": " + "%{y}" +
                                    "<br>"

                                    ,
                                    marker=dict(color=finalDf.target, showscale=True),
                                    mode='markers',
                                    showlegend=False

                                )
                            )

                            fig.update_layout(
                                xaxis={
                                    'title': str(finalDf.columns[skladowe + comb[j][0] + 1])
                                },
                                template=template,
                                yaxis={'title': str(finalDf.columns[skladowe + comb[j][1] + 1])},
                                font=dict(size=plot_font_size))

                            st.plotly_chart(fig, use_container_width=True)
                        st.dataframe(finalDf)
                        def convert_df(df):
                            return df.to_csv(index=False).encode('utf-8')


                        csv = convert_df(finalDf)
                        st.download_button(
                            "Press to Download",
                            csv,
                            "file.csv",
                            "text/csv",
                            key='download-csv'
                        )



        if option1 == 'No':
            przycisk = "Generuj"

            option3 = st.sidebar.selectbox("How many components do you want to use?", list(np.arange(len(df.columns) - 1, 1, -1)))
            template = st.sidebar.selectbox("Select plot style", ("plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn", "simple_white", "none"))
            plot_font_size = st.sidebar.selectbox("Select font size for plots", list(np.arange(10, 31, 1)))
            Gen = st.sidebar.button("Generate")
            if Gen:
                number_of_X_columns = len(df.columns)
                Y_un = None
                X = df.iloc[:, 1:number_of_X_columns].values
                skladowe = option3
                comb = list(combinations(list(range(0, skladowe)), 2))
                X = StandardScaler().fit_transform(X)
                pcaModel = PCA(n_components=skladowe)
                pca = pcaModel.fit(X)
                principalComponents = pca.transform(X)
                st.write("Percentege of variance explained by every PCA: ")
                Pca_var = pd.DataFrame(pca.explained_variance_ratio_.cumsum())
                Pca_var.columns = ["Percentege of variance explained by every PCA: "]
                Pca_var['PCA'] = list(np.arange(1, skladowe + 1))
                st.dataframe(Pca_var)

                fig = px.line(x=list(range(0, skladowe)), y=pca.explained_variance_ratio_, markers=True, labels={
                    "x": "Principal Component",
                    "y": "Variance Explained",

                })

                fig.update_layout(title_text='Scree Plot', title_x=0.5, template=template, font=dict(size=plot_font_size))
                st.plotly_chart(fig, use_container_width=True)
                principalDf = pd.DataFrame(data=principalComponents
                                           )
                finalDf = principalDf
                newColumns = []
                for i in range(len(finalDf.columns)):
                    newColumns.append("PC " + str(i + 1))
                finalDf.columns = newColumns
                X_un = df.columns[1:len(df.columns) ]

                comb = list(combinations(list(range(0, skladowe)), 2))

                CorData = pd.DataFrame(np.transpose(pcaModel.components_))
                CorData.index = X_un
                NewColumns = []

                for i in range(len(CorData.columns)):
                    NewColumns.append("PC_" + str(i + 1))
                CorData.columns = NewColumns


                fig = px.imshow(CorData)
                fig.update_layout(font=dict(size=plot_font_size))
                st.plotly_chart(fig, use_container_width=True)

                for i in range(len(finalDf.columns)):
                    s = principalComponents[:, i]

                    scalex = (1.0 / (s.max() - s.min()))

                    sx = s * scalex

                    finalDf['scaled_PC_' + str(i + 1)] = sx.tolist()

                finalDf['indeks'] = finalDf.index
                new_number_of_X_columns = len(finalDf.columns)
                fig = go.Figure()

                fig.add_trace(

                    go.Scatter(
                        x=finalDf.iloc[:, skladowe + comb[0][0]],
                        y=finalDf.iloc[:, skladowe + comb[0][1]],
                        text=finalDf.indeks.tolist(),
                        hovertemplate=
                        "Index: %{text}" +
                        "<br>" + str(finalDf.columns[skladowe + comb[0][0]]) + ":  " + "%{x}" +
                        "<br>" + str(finalDf.columns[skladowe + comb[0][1]]) + ":  " + "%{y}"

                        ,
                        mode='markers',
                        showlegend=False

                    )
                )
                for i in range(len(X_un)):
                    fig.add_trace(

                        go.Scatter(
                            x=[0, np.transpose(pcaModel.components_)[i][0]],
                            y=[0, np.transpose(pcaModel.components_)[i][1]],
                            text=X_un.tolist(),
                            hovertemplate="%{text}<br>" +
                                          str(finalDf.columns[skladowe + comb[0][0]]) + ":  %{x}<br>" +
                                          str(finalDf.columns[skladowe + comb[0][1]]) + ":  %{y}",
                            mode="lines+markers",
                            showlegend=False
                        )

                    )

                fig.update_layout(
                    title={
                        'text': "Factor Loadings Graph",
                        'y': 0.9,
                        'x': 0.5,
                        'xanchor': 'center',
                        'yanchor': 'top'},
                    template=template,
                    xaxis={
                        'title': str(finalDf.columns[skladowe + comb[0][0]])
                    },
                    yaxis={'title': str(finalDf.columns[skladowe + comb[0][1]])},
                    font=dict(size=plot_font_size)
                )

                st.plotly_chart(fig, use_container_width=True)

                for j in range(len(comb)):
                    fig = go.Figure()

                    fig.add_trace(

                        go.Scatter(
                            x=finalDf.iloc[:, skladowe + comb[j][0]],
                            y=finalDf.iloc[:, skladowe + comb[j][1]],
                            text=finalDf.indeks.tolist(),
                            hovertemplate=
                            "Index: %{text}" +
                            "<br>" + str(finalDf.columns[skladowe + comb[j][0]]) + ": " + "%{x}" +
                            "<br>" + str(finalDf.columns[skladowe + comb[j][1]]) + ": " + "%{y}"

                            ,
                            mode='markers',
                            showlegend=False

                        )
                    )

                    fig.update_layout(
                        xaxis={
                            'title': str(finalDf.columns[skladowe + comb[j][0]])
                        },
                        template=template,
                        yaxis={'title': str(finalDf.columns[skladowe + comb[j][1]])},
                        font=dict(size=plot_font_size)
                    )

                    st.plotly_chart(fig, use_container_width=True)

                st.dataframe(finalDf)


                def convert_df(df):
                    return df.to_csv(index=False).encode('utf-8')


                csv = convert_df(finalDf)
                st.download_button(
                    "Press to Download",
                    csv,
                    "file.csv",
                    "text/csv",
                    key='download-csv'
                )
    if option0=='HCA 2D':
        option1 = st.sidebar.selectbox('Does your dataset contain Y column', ('', 'Yes', 'No'))
        option2= st.sidebar.selectbox("Select color cluster map color",['viridis', 'plasma', 'inferno', 'magma', 'cividis'])

        if option1 == 'Yes':
            X = df.drop(df.columns[len(df.columns) - 1], axis='columns')
            Y = df[df.columns[len(df.columns) - 1]]
        else: X=df
        Gen = st.sidebar.button("Generate")
        if Gen:

            sns.set_theme(color_codes=True)
            number_of_X_columns = len(df.columns) - 1

            X=X.drop(X.columns[0], axis='columns')

            scaler = StandardScaler()
            Z = scaler.fit_transform(X)
            X = pd.DataFrame(Z, columns=X.columns.values)


            g = sns.clustermap(X, figsize=(20, 40), standard_scale=1, cmap=option2)

            st.pyplot(g)
    if option0=="HCA":
        st.subheader("Select columns for HCA")
        dataset=df
        options = st.sidebar.multiselect(
            'What are the columns of your dataset you want to use?',
            list(dataset.columns.values), default=list(dataset.columns.values))

        st.dataframe(dataset[options])
        st.subheader("Enter arguments for dendrogram or leave default")

        title = st.text_input('Title', value='Dendrogram')
        title_font_size = int(st.slider('Title font size', min_value=0, max_value=100))
        title_color = st.color_picker('Title font color', value=None, key=None, help=None, on_change=None, args=None,
                                      kwargs=None)
        ylabel = st.text_input('Y label', value='Distance')
        ylabel_font_size = int(st.slider('Y label font size', min_value=0, max_value=100))
        y_color = st.color_picker('Y label color: ', value=None, key=None, help=None, on_change=None, args=None,
                                  kwargs=None)
        xlabel = st.text_input('X label')
        xlabel_font_size = int(st.slider('X label font size', min_value=0, max_value=100))
        x_color = st.color_picker('X label color', value=None, key=None, help=None, on_change=None, args=None,
                                  kwargs=None)
        method = st.sidebar.selectbox('Which method would you like to use?',
                              options=('single', 'complete', 'average', 'weighted', 'centroid', 'median', 'ward'),
                              index=6)
        st.write('Read more: https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html')
        style = st.sidebar.selectbox('Pick your dendrogram theme',
                             options=['default', 'Solarize_Light2', '_classic_test_patch', '_mpl-gallery',
                                      '_mpl-gallery-nogrid', 'bmh', 'classic', 'fast', 'fivethirtyeight', 'ggplot',
                                      'grayscale', 'seaborn', 'seaborn-bright', 'seaborn-colorblind', 'seaborn-dark',
                                      'seaborn-dark-palette', 'seaborn-darkgrid', 'seaborn-deep', 'seaborn-muted',
                                      'seaborn-notebook', 'seaborn-paper', 'seaborn-pastel', 'seaborn-poster',
                                      'seaborn-talk', 'seaborn-ticks', 'seaborn-white', 'seaborn-whitegrid',
                                      'tableau-colorblind10'], index=0)
        dendrogram_button = st.button('Generate dedrogram')
        st.subheader("Use the dendrogram to find the optimal number of clusters")
        data = dataset[options]
        X = data.values


        def dendrogram_generator(X=X, method=method, ylabel=ylabel, xlabel=xlabel, style=style,
                                 title_font_size=title_font_size, xlabel_font_size=xlabel_font_size,
                                 ylabel_font_size=ylabel_font_size, y_color=y_color, x_color=x_color,
                                 title_color=title_color):
            dendrogram = sch.dendrogram(sch.linkage(X, method=method))
            plt.style.use([style])
            plt.title(title, fontsize=title_font_size, color=title_color)
            plt.ylabel(ylabel, fontsize=ylabel_font_size, color=y_color)
            plt.xlabel(xlabel, fontsize=xlabel_font_size, color=x_color)
            st.pyplot()


        if dendrogram_button:
            dendrogram_generator()

        st.subheader("Measure distance:")
        max_value = st.text_input('Max y value: ')
        if max_value:
            top = int(st.slider('Top line', min_value=0, max_value=int(max_value)))
            bottom = int(st.slider('Bottom line', min_value=0, max_value=int(max_value)))


            def dendrogram_generator_lines(X=X, method=method, ylabel=ylabel, xlabel=xlabel, style=style,
                                           title_font_size=title_font_size, xlabel_font_size=xlabel_font_size,
                                           ylabel_font_size=ylabel_font_size, y_color=y_color, x_color=x_color,
                                           title_color=title_color, top_line=top, bottom_line=bottom):
                dendrogram = sch.dendrogram(sch.linkage(X, method=method))
                plt.style.use([style])
                plt.title(title, fontsize=title_font_size, color=title_color)
                plt.ylabel(ylabel, fontsize=ylabel_font_size, color=y_color)
                plt.xlabel(xlabel, fontsize=xlabel_font_size, color=x_color)
                plt.axhline(y=top_line, color='black', linestyle='-')
                plt.axhline(y=bottom_line, color='black', linestyle='-')
                st.pyplot()


            dendrogram_generator_lines()
            st.write("The distance between bottom and top line is equal to: ", str(abs(top - bottom)))

        st.subheader("Clusterization")
        num_of_clusters = int(st.number_input("Number of clusters you wanna use:", min_value=1, step=1))
        affinity = st.selectbox('Metric used to compute the linkage.',
                                options=["euclidean", "l1", "l2", "manhattan", "cosine", "precomputed"])
        linkage = st.selectbox(
            'Which linkage criterion to use. The linkage criterion determines which distance to use between sets of observation. The algorithm will merge the pairs of cluster that minimize this criterion.',
            options=['ward', 'complete', 'average', 'single'])
        if affinity == "euclidean":
            linkage = 'ward'
        hc = AgglomerativeClustering(n_clusters=num_of_clusters, affinity=affinity, linkage=linkage)
        y_hc = hc.fit_predict(X)

        dataset['Clusters'] = y_hc
        st.dataframe(dataset)


        @st.cache
        def convert_df(df):
            return df.to_csv(index=False).encode('utf-8')


        csv = convert_df(dataset)

        st.download_button(
            "Press to Download",
            csv,
            "file.csv",
            "text/csv",
            key='download-csv'
        )