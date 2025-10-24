%% Setup
addpath(genpath(pwd)) 


%Lista de datasets a ejecutar (deben estar colocados en dataset_papers/mat

lista_datasets_artificiales = {
%     'ac_10clusters_2vars',
%     'ac_10clusters_3vars',
%     'ac_10clusters_5vars',
%     'ac_12clusters_2vars',
%     'ac_12clusters_7vars',
%     'ac_5clusters_3vars',
%     'ac_6clusters_7vars',
%     'ac_7clusters_2vars',
%     'ac_7clusters_3vars',
%     'ac_7clusters_5vars',
%     'data_balls_10clusters_12_vars_2',
%     'data_balls_15clusters_20_vars_1',
%     'data_balls_5clusters_5_vars_3',
%     'data_balls_5clusters_6_vars_1',
%     'data_balls_6clusters_10_vars_5',
%     'data_balls_6clusters_8_vars_2',
%     'data_balls_7clusters_7_vars_3',
%     'data_balls_7clusters_9_vars_4',
%     'data_balls_8clusters_10_vars_1',
% %     'data_balls_9clusters_9_vars_3',
%     'data_corners_6clusters_1',
%     'data_corners_6clusters_2',
%     'data_corners_7clusters_1',
%     'data_corners_7clusters_2'
%     'data_corners_10clusters_1',
%     'data_corners_10clusters_3',
%     'data_corners_12clusters_1',
%     'data_corners_12clusters_3',
%     'data_corners_14clusters_1',
%     'data_corners_14clusters_3',
};

lista_datasets_reales = {
%     'MSRA25'
% 'accent_recognition'
% 'coil20'
% 'dermatology'
% 'ecoli'
% 'GLIOMA'
% 'glass_identification'
% 'htru2'
'ionosphere'
% 'iris'
% 'lung'
% 'mfeat-fou'
% 'mfeat-pix'
% 'mice'
% % 'movement_libras'
% 'newthyroid'
% 'optdigits'
% 'orlraws10P'
% 'parkinsons'
% 'pendigits'
% 'rice'
% 'seeds'
% 'semeion'
% 'TOX_171'
% 'texture'
% 'twonorm'
% 'user_modeling'
% 'wdbc'
% 'wifi_localization'
% 'Yale'

};

%Creo un diccionario para almacenar ID:NombreMetodo
ids = [23,1,16,14];
nombres = {'INFFS2020','INFFS','RNE','FMIUFS'};

dicc_id_nombre = containers.Map(ids,nombres);

for j = 1:4
    for i = 1:length(lista_datasets_reales)
        id = ids(j);   
        nombre_dataset = lista_datasets_reales{i};
        clear Result
        clc
        
        ruta = ['../datasets_papers/mat/',nombre_dataset, '_X.mat'];
        disp(['Ruta actual: ', ruta])
        load(ruta)

        Selection_Method = UFSNaming_mod(id);

        Result = Auto_UFSTool(double(X),Selection_Method); % double para evitar problemas 
        ruta_guardado_Result = ['../results_papers/',nombre_dataset,'/result',dicc_id_nombre(id),'_',nombre_dataset,'.mat'];
        save(ruta_guardado_Result, 'Result');

    end
end



