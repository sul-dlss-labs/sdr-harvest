# Updating Solr Configs

## Create tunnel
```
ssh -N -L 8983:sul-solr-test.stanford.edu:443 semantic-search-demo.stanford.edu
```

## Upload updated configs
```
curl --resolve sul-solr-test.stanford.edu:8983:127.0.0.1 \
  -X PUT \
  --header "Content-Type:application/octet-stream" \
  --data-binary @solr_configs/solrconfig.xml \
  "https://sul-solr-test.stanford.edu:8983/solr/api/configsets/semantic-search/solrconfig.xml"
```

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
