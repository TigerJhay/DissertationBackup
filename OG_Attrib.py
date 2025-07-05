def attrib_table(temp_df_attrib):
    #--------------------------------------------------------------------------------------------
    #Extracting phrases for creating corpora that will be use in data visualization
    # FF: temp_df_attrib here is a cleaned dataset came from datacleaning function
    #--------------------------------------------------------------------------------------------
    df_reviews = temp_df_attrib.drop(axis=1, columns=["Date"])
    df = pd.DataFrame()
    def extract_attrib(attrib_value):
        df_temp = df_reviews.loc[df_reviews["Reviews"].str.contains(attrib_value, regex=False)]
        df_temp["Reviews"] = df_temp["Reviews"].str.replace('[0-9]', "", regex=True)
        
        if attrib_value == "battery":
            df_temp["Reviews"] = df_temp["Reviews"].str.extract(r'\b((?:\w+\W+){0,2}battery\b(?:\W+\w+){0,2})')
        elif attrib_value == "speed":
            df_temp["Reviews"] = df_temp["Reviews"].str.extract(r'\b((?:\w+\W+){0,2}speed\b(?:\W+\w+){0,2})')
        elif attrib_value == "memory":
            df_temp["Reviews"] = df_temp["Reviews"].str.extract(r'\b((?:\w+\W+){0,2}memory\b(?:\W+\w+){0,2})')
        elif attrib_value == "screen":
            df_temp["Reviews"] = df_temp["Reviews"].str.extract(r'\b((?:\w+\W+){0,2}screen\b(?:\W+\w+){0,2})')
        elif attrib_value == "screen":
            df_temp["Reviews"] = df_temp["Reviews"].str.extract(r'\b((?:\w+\W+){0,2}audio\b(?:\W+\w+){0,2})')
        else:
            df_temp["Attribute"] = attrib_value
        
        df_temp = df_temp.dropna(axis=0, subset=["Reviews"], how='any')
        df_temp = df_temp.drop_duplicates(subset="Reviews")
        df_temp["Attribute"] = attrib_value
        return df_temp

    list_attrib = ["battery", "screen", "speed", "memory", "audio"]
    for attrib in list_attrib:
        df = pd.concat([df, extract_attrib(attrib)])

    attrib_matrix = pd.DataFrame(columns=["Model", "Batt_PR","Batt_NR", "Scr_PR", "Scr_NR", "Spd_PR", "Spd_NR", "Mem_PR", "Mem_NR", "Aud_PR", "Aud_NR"])
    gadget_list = df_reviews["Model"].unique()

    def convert_to_matrix(gadget_model):
        df_model = df.loc[df["Model"].str.contains(gadget_model)]
        df_rev = df_model.loc[df_model["Reviews"].str.contains("battery")]
        batt_rpos = df_rev["Rating"].value_counts().get(1,0)
        batt_rneg = df_rev["Rating"].value_counts().get(0,0)

        df_rev = df_model.loc[df_model["Reviews"].str.contains("screen")]
        scr_rpos = df_rev["Rating"].value_counts().get(1,0)
        scr_rneg = df_rev["Rating"].value_counts().get(0,0)

        df_rev = df_model.loc[df_model["Reviews"].str.contains("speed")]
        spd_rpos = df_rev["Rating"].value_counts().get(1,0)
        spd_rneg = df_rev["Rating"].value_counts().get(0,0)

        df_rev = df_model.loc[df_model["Reviews"].str.contains("memory")]
        mem_rpos = df_rev["Rating"].value_counts().get(1,0)
        mem_rneg = df_rev["Rating"].value_counts().get(0,0)
        
        df_rev = df_model.loc[df_model["Reviews"].str.contains("audio")]
        aud_rpos = df_rev["Rating"].value_counts().get(1,0)
        aud_rneg = df_rev["Rating"].value_counts().get(0,0)

        row_value = [gadget_model, batt_rpos, batt_rneg, scr_rpos, scr_rneg, spd_rpos, spd_rneg, mem_rpos, mem_rneg, aud_rpos, aud_rneg]    
        return row_value

    for colname in gadget_list:
        attrib_matrix.loc[len(attrib_matrix)] = convert_to_matrix(colname)
    attrib_matrix.to_sql(con=sqlengine, name="attribute_table", if_exists='replace', index=True)