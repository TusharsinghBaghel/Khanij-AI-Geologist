from groq import Groq
client = Groq(
    api_key="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",
)

chat_completion = client.chat.completions.create(
    #
    # Required parameters
    #
    messages=[
        # Set an optional system message. This sets the behavior of the
        # assistant and can be used to provide specific instructions for
        # how it should behave throughout the conversation.
        {
            "role": "system",
            "content": '''You are Khanij, an AI assistant for MECL (Mineral Exploration and Consultancy Limited). Based on the user query, determine the appropriate task to perform:
                        
                        - Print "Heatmap": If the query is related to creating a heatmap.
                          In the next line, print a Python list with one element: the chemical formula in lowercase mentioned in the query.
                          - If the query mentions geophysical magnetic data, add 'magnetic_a' to the list.
                          - If it's related to gravity, add 'bouguer_an' to the list.
                          - If it's related to elevation, add 'elevation_' to the list.
                          - Do not print anything else.

                        - Print "Stat": If the query is related to finding mean, mode, variance, etc., of data.
                          In the next line, print a Python list with one element: the chemical formula in lowercase mentioned in the query.

                        - Print "Contour": If the query is related to creating contour maps.
                          In the next line, print a Python list with one element: the chemical formula in lowercase mentioned in the query.
                          - If the query mentions geophysical magnetic data, add 'magnetic_a' to the list.
                          - If it's related to gravity, add 'bouguer_an' to the list.
                          - If it's related to elevation, add 'elevation_' to the list.
                          - Do not print anything else.

                        - Print "Exploration": If the query is about known exploration or excavation sites of the region.

                        - Print "KrigingInterpolation": If the query is related to Kriging interpolation.
                          In the next line, print a Python list with one element: the chemical formula in lowercase mentioned in the query.
                          - If the query mentions geophysical magnetic data, add 'magnetic_a' to the list.
                          - If it's related to gravity, add 'bouguer_an' to the list.
                          - If it's related to elevation, add 'elevation_' to the list.
                          - Do not print anything else.

                        - Print "IDWInterpolation": If the query is related to IDW (Inverse Distance Weighting) interpolation.
                          In the next line, print a Python list with one element: the chemical formula in lowercase mentioned in the query.
                          - If the query mentions geophysical magnetic data, add 'magnetic_a' to the list.
                          - If it's related to gravity, add 'bouguer_an' to the list.
                          - If it's related to elevation, add 'elevation_' to the list.
                          - Do not print anything else.

                        - Print "SplineInterpolation": If the query is related to spline interpolation.
                          In the next line, print a Python list with one element: the chemical formula in lowercase mentioned in the query.
                          - If the query mentions geophysical magnetic data, add 'magnetic_a' to the list.
                          - If it's related to gravity, add 'bouguer_an' to the list.
                          - If it's related to elevation, add 'elevation_' to the list.
                          - Do not print anything else.

                        - Print "NNInterpolation": If the query is related to nearest neighbor interpolation.
                          In the next line, print a Python list with one element: the chemical formula in lowercase mentioned in the query.
                          - If the query mentions geophysical magnetic data, add 'magnetic_a' to the list.
                          - If it's related to gravity, add 'bouguer_an' to the list.
                          - If it's related to elevation, add 'elevation_' to the list.
                          - Do not print anything else.

                        - Print "Histogram": If the query is related to creating a histogram.
                          In the next line, print a Python list of chemical formulas mentioned in the query.
                          - If there are no chemicals mentioned, print a list containing the string 'all'.

                        - Print a response according to your knowledge: If the query does not relate to any of the specified tasks.
                        '''
        },
        # Set a user message for the assistant to respond to.
        {
            "role": "user",
            "content": "Create a heatmap of magnetic data",
        }
    ],

    # The language model which will generate the completion.
    model="llama3-70b-8192",

    #
    # Optional parameters
    #

    # Controls randomness: lowering results in less random completions.
    # As the temperature approaches zero, the model will become deterministic
    # and repetitive.
    temperature=0.5,

    # The maximum number of tokens to generate. Requests can use up to
    # 2048 tokens shared between prompt and completion.
    max_tokens=1024,

    # Controls diversity via nucleus sampling: 0.5 means half of all
    # likelihood-weighted options are considered.
    top_p=1,

    # A stop sequence is a predefined or user-specified text string that
    # signals an AI to stop generating content, ensuring its responses
    # remain focused and concise. Examples include punctuation marks and
    # markers like "[end]".
    stop=None,

    # If set, partial message deltas will be sent.
    stream=False,
)
print(chat_completion.choices[0].message.content)














if method==1:
            #IDW Interpolation
            interpolation_type = "IDW"
            grid_z = griddata(points, values, (grid_lat, grid_long), method='cubic')
        elif method==2:
            #Spline Interpolation:
            interpolation_type = "Spline"
            tck = bisplrep(points[:, 0], points[:, 1], values, s=0)
            #s is the smoothing value and is chosen by trial and error
            grid_z = bisplev(grid_lat[:, 0], grid_long[0, :], tck)
        elif method==3:
            #Kriging interpolation
            interpolation_type = "Kriging"
            OK = OrdinaryKriging(points[:, 0], points[:, 1], values, variogram_model='linear')
            grid_z, _ = OK.execute('grid', grid_lat[0, :], grid_long[:, 0])
            
        elif method==4:
            interpolation_type = "Nearest Neighbor"
            grid_z = griddata(points, values, (grid_lat, grid_long), method='nearest')