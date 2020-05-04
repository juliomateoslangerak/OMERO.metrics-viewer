
def get_tables(omero_object, namespace_start='', name_filter=''):
    tables_list = list()
    resources = omero_object._conn.getSharedResources()
    for ann in omero_object.listAnnotations():
        if isinstance(ann, gw.FileAnnotationWrapper) and \
                ann.getNs().startswith(namespace_start) and \
                name_filter in ann.getFileName():
            table_file = omero_object._conn.getObject("OriginalFile", attributes={'name': ann.getFileName()})
            table = resources.openTable(table_file._obj)
            tables_list.append(table)

    return tables_list

