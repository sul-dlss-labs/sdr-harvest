# Updating Solr Configs

## Create tunnel
```
ssh -N -L 8983:sul-solr-test.stanford.edu:443 semantic-search-demo.stanford.edu
```

## Upload updated configs
Use the v1 Configsets API; the v2 `/solr/api/configsets/...` path returns 404 on
this server. Upload `schema.xml` before `solrconfig.xml`, and run the loop
from the repository root.

```
for file in schema.xml solrconfig.xml; do
  curl -u USERNAME:PASSWORD \
    --resolve sul-solr-test.stanford.edu:8983:127.0.0.1 \
    -X POST \
    --header "Content-Type:application/octet-stream" \
    --data-binary @solr_configs/$file \
    "https://sul-solr-test.stanford.edu:8983/solr/admin/configs?action=UPLOAD&name=semantic-search&filePath=$file&overwrite=true"
done
```

`solrconfig.xml` sets `ClassicIndexSchemaFactory`, so Solr reads `schema.xml`
directly and the Schema API cannot change fields. Make schema changes in
`schema.xml` and upload it; a reload is required for either file to take effect.

## Reload the collection
You can also do this via the Solr admin UI. 
[https://sul-solr-test.stanford.edu/solr/#/~collections/contracts-test](https://sul-solr-test.stanford.edu/solr/#/~collections/contracts-test)

```
curl -u USERNAME:PASSWORD \
  --resolve sul-solr-test.stanford.edu:8983:127.0.0.1 \
  "https://sul-solr-test.stanford.edu:8983/solr/admin/collections?action=RELOAD&name=semantic-search-demo"
```

## Verify

See if the collection was reloaded successfully.
```
curl -u USERNAME:PASSWORD \
  --resolve sul-solr-test.stanford.edu:8983:127.0.0.1 \
  "https://sul-solr-test.stanford.edu:8983/solr/admin/collections?action=CLUSTERSTATUS&collection=semantic-search-demo"
```

Confirm a schema change is live, for example a dynamic field:
```
curl -u USERNAME:PASSWORD \
  --resolve sul-solr-test.stanford.edu:8983:127.0.0.1 \
  "https://sul-solr-test.stanford.edu:8983/solr/semantic-search-demo/schema/dynamicfields/*_ism"
```

Check to see if the hybrid handler is available.
```
curl --resolve sul-solr-test.stanford.edu:8983:127.0.0.1 \
-X POST \
-H "Content-Type: application/json" \
"https://sul-solr-test.stanford.edu:8983/solr/semantic-search-demo/hybrid" \
-d '{
  "queries": {
    "lexical1": { "lucene": { "query": "title_tesi:stanford" } },
    "lexical2": { "lucene": { "query": "title_tesi:history" } }
  },
  "limit": 10,
  "fields": ["id", "score"],
  "params": {
    "combiner": true,
    "combiner.query": ["lexical1", "lexical2"]
  }
}'
```
