for file in *tar.gz
do
    folder_name=`echo $file | sed "s/.tar.gz//g"`
    tar -xzf $file
    rm -r ../deepcc.v1.0/rl-module/$folder_name
    mv $folder_name ../deepcc.v1.0/rl-module/
done

