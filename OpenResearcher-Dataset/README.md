---
dataset_info:
- config_name: seed_42
  features:
  - name: qid
    dtype: int64
  - name: question
    dtype: string
  - name: answer
    dtype: string
  - name: messages
    list:
    - name: channel
      dtype: string
    - name: content
      list:
      - name: channel_config
        struct:
        - name: channel_required
          dtype: bool
        - name: valid_channels
          list: string
      - name: conversation_start_date
        dtype: string
      - name: knowledge_cutoff
        dtype: string
      - name: model_identity
        dtype: string
      - name: reasoning_effort
        dtype: string
      - name: text
        dtype: string
      - name: tools
        struct:
        - name: browser
          struct:
          - name: description
            dtype: string
          - name: name
            dtype: string
          - name: tools
            list:
            - name: description
              dtype: string
            - name: name
              dtype: string
            - name: parameters
              struct:
              - name: properties
                struct:
                - name: cursor
                  struct:
                  - name: default
                    dtype: int64
                  - name: type
                    dtype: string
                - name: id
                  struct:
                  - name: default
                    dtype: int64
                  - name: type
                    list: string
                - name: loc
                  struct:
                  - name: default
                    dtype: int64
                  - name: type
                    dtype: string
                - name: num_lines
                  struct:
                  - name: default
                    dtype: int64
                  - name: type
                    dtype: string
                - name: pattern
                  struct:
                  - name: type
                    dtype: string
                - name: query
                  struct:
                  - name: type
                    dtype: string
                - name: source
                  struct:
                  - name: type
                    dtype: string
                - name: topn
                  struct:
                  - name: default
                    dtype: int64
                  - name: type
                    dtype: string
                - name: view_source
                  struct:
                  - name: default
                    dtype: bool
                  - name: type
                    dtype: string
              - name: required
                list: string
              - name: type
                dtype: string
      - name: type
        dtype: string
    - name: content_type
      dtype: string
    - name: name
      dtype: string
    - name: recipient
      dtype: string
    - name: role
      dtype: string
  - name: latency_s
    dtype: float64
  - name: error
    dtype: 'null'
  - name: attempts
    dtype: int64
  - name: status
    dtype: string
  - name: chunk_idx
    dtype: int64
  - name: num_chunks
    dtype: int64
  splits:
  - name: train
    num_bytes: 1117982919
    num_examples: 6102
  download_size: 461369938
  dataset_size: 1117982919
- config_name: seed_43
  features:
  - name: qid
    dtype: int64
  - name: question
    dtype: string
  - name: answer
    dtype: string
  - name: messages
    list:
    - name: channel
      dtype: string
    - name: content
      list:
      - name: channel_config
        struct:
        - name: channel_required
          dtype: bool
        - name: valid_channels
          list: string
      - name: conversation_start_date
        dtype: string
      - name: knowledge_cutoff
        dtype: string
      - name: model_identity
        dtype: string
      - name: reasoning_effort
        dtype: string
      - name: text
        dtype: string
      - name: tools
        struct:
        - name: browser
          struct:
          - name: description
            dtype: string
          - name: name
            dtype: string
          - name: tools
            list:
            - name: description
              dtype: string
            - name: name
              dtype: string
            - name: parameters
              struct:
              - name: properties
                struct:
                - name: cursor
                  struct:
                  - name: default
                    dtype: int64
                  - name: type
                    dtype: string
                - name: id
                  struct:
                  - name: default
                    dtype: int64
                  - name: type
                    list: string
                - name: loc
                  struct:
                  - name: default
                    dtype: int64
                  - name: type
                    dtype: string
                - name: num_lines
                  struct:
                  - name: default
                    dtype: int64
                  - name: type
                    dtype: string
                - name: pattern
                  struct:
                  - name: type
                    dtype: string
                - name: query
                  struct:
                  - name: type
                    dtype: string
                - name: source
                  struct:
                  - name: type
                    dtype: string
                - name: topn
                  struct:
                  - name: default
                    dtype: int64
                  - name: type
                    dtype: string
                - name: view_source
                  struct:
                  - name: default
                    dtype: bool
                  - name: type
                    dtype: string
              - name: required
                list: string
              - name: type
                dtype: string
      - name: type
        dtype: string
    - name: content_type
      dtype: string
    - name: name
      dtype: string
    - name: recipient
      dtype: string
    - name: role
      dtype: string
  - name: latency_s
    dtype: float64
  - name: error
    dtype: 'null'
  - name: attempts
    dtype: int64
  - name: status
    dtype: string
  - name: chunk_idx
    dtype: int64
  - name: num_chunks
    dtype: int64
  splits:
  - name: train
    num_bytes: 1158931340
    num_examples: 6102
  download_size: 477917292
  dataset_size: 1158931340
- config_name: seed_44
  features:
  - name: qid
    dtype: int64
  - name: question
    dtype: string
  - name: answer
    dtype: string
  - name: messages
    list:
    - name: channel
      dtype: string
    - name: content
      list:
      - name: channel_config
        struct:
        - name: channel_required
          dtype: bool
        - name: valid_channels
          list: string
      - name: conversation_start_date
        dtype: string
      - name: knowledge_cutoff
        dtype: string
      - name: model_identity
        dtype: string
      - name: reasoning_effort
        dtype: string
      - name: text
        dtype: string
      - name: tools
        struct:
        - name: browser
          struct:
          - name: description
            dtype: string
          - name: name
            dtype: string
          - name: tools
            list:
            - name: description
              dtype: string
            - name: name
              dtype: string
            - name: parameters
              struct:
              - name: properties
                struct:
                - name: cursor
                  struct:
                  - name: default
                    dtype: int64
                  - name: type
                    dtype: string
                - name: id
                  struct:
                  - name: default
                    dtype: int64
                  - name: type
                    list: string
                - name: loc
                  struct:
                  - name: default
                    dtype: int64
                  - name: type
                    dtype: string
                - name: num_lines
                  struct:
                  - name: default
                    dtype: int64
                  - name: type
                    dtype: string
                - name: pattern
                  struct:
                  - name: type
                    dtype: string
                - name: query
                  struct:
                  - name: type
                    dtype: string
                - name: source
                  struct:
                  - name: type
                    dtype: string
                - name: topn
                  struct:
                  - name: default
                    dtype: int64
                  - name: type
                    dtype: string
                - name: view_source
                  struct:
                  - name: default
                    dtype: bool
                  - name: type
                    dtype: string
              - name: required
                list: string
              - name: type
                dtype: string
      - name: type
        dtype: string
    - name: content_type
      dtype: string
    - name: name
      dtype: string
    - name: recipient
      dtype: string
    - name: role
      dtype: string
  - name: latency_s
    dtype: float64
  - name: error
    dtype: string
  - name: attempts
    dtype: int64
  - name: status
    dtype: string
  - name: chunk_idx
    dtype: int64
  - name: num_chunks
    dtype: int64
  splits:
  - name: train
    num_bytes: 1108145493
    num_examples: 6102
  download_size: 456027149
  dataset_size: 1108145493
- config_name: seed_45
  features:
  - name: qid
    dtype: int64
  - name: question
    dtype: string
  - name: answer
    dtype: string
  - name: messages
    list:
    - name: channel
      dtype: string
    - name: content
      list:
      - name: channel_config
        struct:
        - name: channel_required
          dtype: bool
        - name: valid_channels
          list: string
      - name: conversation_start_date
        dtype: string
      - name: knowledge_cutoff
        dtype: string
      - name: model_identity
        dtype: string
      - name: reasoning_effort
        dtype: string
      - name: text
        dtype: string
      - name: tools
        struct:
        - name: browser
          struct:
          - name: description
            dtype: string
          - name: name
            dtype: string
          - name: tools
            list:
            - name: description
              dtype: string
            - name: name
              dtype: string
            - name: parameters
              struct:
              - name: properties
                struct:
                - name: cursor
                  struct:
                  - name: default
                    dtype: int64
                  - name: type
                    dtype: string
                - name: id
                  struct:
                  - name: default
                    dtype: int64
                  - name: type
                    list: string
                - name: loc
                  struct:
                  - name: default
                    dtype: int64
                  - name: type
                    dtype: string
                - name: num_lines
                  struct:
                  - name: default
                    dtype: int64
                  - name: type
                    dtype: string
                - name: pattern
                  struct:
                  - name: type
                    dtype: string
                - name: query
                  struct:
                  - name: type
                    dtype: string
                - name: source
                  struct:
                  - name: type
                    dtype: string
                - name: topn
                  struct:
                  - name: default
                    dtype: int64
                  - name: type
                    dtype: string
                - name: view_source
                  struct:
                  - name: default
                    dtype: bool
                  - name: type
                    dtype: string
              - name: required
                list: string
              - name: type
                dtype: string
      - name: type
        dtype: string
    - name: content_type
      dtype: string
    - name: name
      dtype: string
    - name: recipient
      dtype: string
    - name: role
      dtype: string
  - name: latency_s
    dtype: float64
  - name: error
    dtype: 'null'
  - name: attempts
    dtype: int64
  - name: status
    dtype: string
  - name: chunk_idx
    dtype: int64
  - name: num_chunks
    dtype: int64
  splits:
  - name: train
    num_bytes: 1106900749
    num_examples: 6102
  download_size: 455271833
  dataset_size: 1106900749
- config_name: seed_46
  features:
  - name: qid
    dtype: int64
  - name: question
    dtype: string
  - name: answer
    dtype: string
  - name: messages
    list:
    - name: channel
      dtype: string
    - name: content
      list:
      - name: channel_config
        struct:
        - name: channel_required
          dtype: bool
        - name: valid_channels
          list: string
      - name: conversation_start_date
        dtype: string
      - name: knowledge_cutoff
        dtype: string
      - name: model_identity
        dtype: string
      - name: reasoning_effort
        dtype: string
      - name: text
        dtype: string
      - name: tools
        struct:
        - name: browser
          struct:
          - name: description
            dtype: string
          - name: name
            dtype: string
          - name: tools
            list:
            - name: description
              dtype: string
            - name: name
              dtype: string
            - name: parameters
              struct:
              - name: properties
                struct:
                - name: cursor
                  struct:
                  - name: default
                    dtype: int64
                  - name: type
                    dtype: string
                - name: id
                  struct:
                  - name: default
                    dtype: int64
                  - name: type
                    list: string
                - name: loc
                  struct:
                  - name: default
                    dtype: int64
                  - name: type
                    dtype: string
                - name: num_lines
                  struct:
                  - name: default
                    dtype: int64
                  - name: type
                    dtype: string
                - name: pattern
                  struct:
                  - name: type
                    dtype: string
                - name: query
                  struct:
                  - name: type
                    dtype: string
                - name: source
                  struct:
                  - name: type
                    dtype: string
                - name: topn
                  struct:
                  - name: default
                    dtype: int64
                  - name: type
                    dtype: string
                - name: view_source
                  struct:
                  - name: default
                    dtype: bool
                  - name: type
                    dtype: string
              - name: required
                list: string
              - name: type
                dtype: string
      - name: type
        dtype: string
    - name: content_type
      dtype: string
    - name: name
      dtype: string
    - name: recipient
      dtype: string
    - name: role
      dtype: string
  - name: latency_s
    dtype: float64
  - name: error
    dtype: 'null'
  - name: attempts
    dtype: int64
  - name: status
    dtype: string
  - name: chunk_idx
    dtype: int64
  - name: num_chunks
    dtype: int64
  splits:
  - name: train
    num_bytes: 1135517221
    num_examples: 6102
  download_size: 468875734
  dataset_size: 1135517221
- config_name: seed_47
  features:
  - name: qid
    dtype: int64
  - name: question
    dtype: string
  - name: answer
    dtype: string
  - name: messages
    list:
    - name: channel
      dtype: string
    - name: content
      list:
      - name: channel_config
        struct:
        - name: channel_required
          dtype: bool
        - name: valid_channels
          list: string
      - name: conversation_start_date
        dtype: string
      - name: knowledge_cutoff
        dtype: string
      - name: model_identity
        dtype: string
      - name: reasoning_effort
        dtype: string
      - name: text
        dtype: string
      - name: tools
        struct:
        - name: browser
          struct:
          - name: description
            dtype: string
          - name: name
            dtype: string
          - name: tools
            list:
            - name: description
              dtype: string
            - name: name
              dtype: string
            - name: parameters
              struct:
              - name: properties
                struct:
                - name: cursor
                  struct:
                  - name: default
                    dtype: int64
                  - name: type
                    dtype: string
                - name: id
                  struct:
                  - name: default
                    dtype: int64
                  - name: type
                    list: string
                - name: loc
                  struct:
                  - name: default
                    dtype: int64
                  - name: type
                    dtype: string
                - name: num_lines
                  struct:
                  - name: default
                    dtype: int64
                  - name: type
                    dtype: string
                - name: pattern
                  struct:
                  - name: type
                    dtype: string
                - name: query
                  struct:
                  - name: type
                    dtype: string
                - name: source
                  struct:
                  - name: type
                    dtype: string
                - name: topn
                  struct:
                  - name: default
                    dtype: int64
                  - name: type
                    dtype: string
                - name: view_source
                  struct:
                  - name: default
                    dtype: bool
                  - name: type
                    dtype: string
              - name: required
                list: string
              - name: type
                dtype: string
      - name: type
        dtype: string
    - name: content_type
      dtype: string
    - name: name
      dtype: string
    - name: recipient
      dtype: string
    - name: role
      dtype: string
  - name: latency_s
    dtype: float64
  - name: error
    dtype: 'null'
  - name: attempts
    dtype: int64
  - name: status
    dtype: string
  - name: chunk_idx
    dtype: int64
  - name: num_chunks
    dtype: int64
  splits:
  - name: train
    num_bytes: 1179953502
    num_examples: 6102
  download_size: 487937315
  dataset_size: 1179953502
- config_name: seed_48
  features:
  - name: qid
    dtype: int64
  - name: question
    dtype: string
  - name: answer
    dtype: string
  - name: messages
    list:
    - name: channel
      dtype: string
    - name: content
      list:
      - name: channel_config
        struct:
        - name: channel_required
          dtype: bool
        - name: valid_channels
          list: string
      - name: conversation_start_date
        dtype: string
      - name: knowledge_cutoff
        dtype: string
      - name: model_identity
        dtype: string
      - name: reasoning_effort
        dtype: string
      - name: text
        dtype: string
      - name: tools
        struct:
        - name: browser
          struct:
          - name: description
            dtype: string
          - name: name
            dtype: string
          - name: tools
            list:
            - name: description
              dtype: string
            - name: name
              dtype: string
            - name: parameters
              struct:
              - name: properties
                struct:
                - name: cursor
                  struct:
                  - name: default
                    dtype: int64
                  - name: type
                    dtype: string
                - name: id
                  struct:
                  - name: default
                    dtype: int64
                  - name: type
                    list: string
                - name: loc
                  struct:
                  - name: default
                    dtype: int64
                  - name: type
                    dtype: string
                - name: num_lines
                  struct:
                  - name: default
                    dtype: int64
                  - name: type
                    dtype: string
                - name: pattern
                  struct:
                  - name: type
                    dtype: string
                - name: query
                  struct:
                  - name: type
                    dtype: string
                - name: source
                  struct:
                  - name: type
                    dtype: string
                - name: topn
                  struct:
                  - name: default
                    dtype: int64
                  - name: type
                    dtype: string
                - name: view_source
                  struct:
                  - name: default
                    dtype: bool
                  - name: type
                    dtype: string
              - name: required
                list: string
              - name: type
                dtype: string
      - name: type
        dtype: string
    - name: content_type
      dtype: string
    - name: name
      dtype: string
    - name: recipient
      dtype: string
    - name: role
      dtype: string
  - name: latency_s
    dtype: float64
  - name: error
    dtype: 'null'
  - name: attempts
    dtype: int64
  - name: status
    dtype: string
  - name: chunk_idx
    dtype: int64
  - name: num_chunks
    dtype: int64
  splits:
  - name: train
    num_bytes: 1171147444
    num_examples: 6102
  download_size: 483010306
  dataset_size: 1171147444
- config_name: seed_49
  features:
  - name: qid
    dtype: int64
  - name: question
    dtype: string
  - name: answer
    dtype: string
  - name: messages
    list:
    - name: channel
      dtype: string
    - name: content
      list:
      - name: channel_config
        struct:
        - name: channel_required
          dtype: bool
        - name: valid_channels
          list: string
      - name: conversation_start_date
        dtype: string
      - name: knowledge_cutoff
        dtype: string
      - name: model_identity
        dtype: string
      - name: reasoning_effort
        dtype: string
      - name: text
        dtype: string
      - name: tools
        struct:
        - name: browser
          struct:
          - name: description
            dtype: string
          - name: name
            dtype: string
          - name: tools
            list:
            - name: description
              dtype: string
            - name: name
              dtype: string
            - name: parameters
              struct:
              - name: properties
                struct:
                - name: cursor
                  struct:
                  - name: default
                    dtype: int64
                  - name: type
                    dtype: string
                - name: id
                  struct:
                  - name: default
                    dtype: int64
                  - name: type
                    list: string
                - name: loc
                  struct:
                  - name: default
                    dtype: int64
                  - name: type
                    dtype: string
                - name: num_lines
                  struct:
                  - name: default
                    dtype: int64
                  - name: type
                    dtype: string
                - name: pattern
                  struct:
                  - name: type
                    dtype: string
                - name: query
                  struct:
                  - name: type
                    dtype: string
                - name: source
                  struct:
                  - name: type
                    dtype: string
                - name: topn
                  struct:
                  - name: default
                    dtype: int64
                  - name: type
                    dtype: string
                - name: view_source
                  struct:
                  - name: default
                    dtype: bool
                  - name: type
                    dtype: string
              - name: required
                list: string
              - name: type
                dtype: string
      - name: type
        dtype: string
    - name: content_type
      dtype: string
    - name: name
      dtype: string
    - name: recipient
      dtype: string
    - name: role
      dtype: string
  - name: latency_s
    dtype: float64
  - name: error
    dtype: 'null'
  - name: attempts
    dtype: int64
  - name: status
    dtype: string
  - name: chunk_idx
    dtype: int64
  - name: num_chunks
    dtype: int64
  splits:
  - name: train
    num_bytes: 1120741628
    num_examples: 6102
  download_size: 461516938
  dataset_size: 1120741628
- config_name: seed_50
  features:
  - name: qid
    dtype: int64
  - name: question
    dtype: string
  - name: answer
    dtype: string
  - name: messages
    list:
    - name: channel
      dtype: string
    - name: content
      list:
      - name: channel_config
        struct:
        - name: channel_required
          dtype: bool
        - name: valid_channels
          list: string
      - name: conversation_start_date
        dtype: string
      - name: knowledge_cutoff
        dtype: string
      - name: model_identity
        dtype: string
      - name: reasoning_effort
        dtype: string
      - name: text
        dtype: string
      - name: tools
        struct:
        - name: browser
          struct:
          - name: description
            dtype: string
          - name: name
            dtype: string
          - name: tools
            list:
            - name: description
              dtype: string
            - name: name
              dtype: string
            - name: parameters
              struct:
              - name: properties
                struct:
                - name: cursor
                  struct:
                  - name: default
                    dtype: int64
                  - name: type
                    dtype: string
                - name: id
                  struct:
                  - name: default
                    dtype: int64
                  - name: type
                    list: string
                - name: loc
                  struct:
                  - name: default
                    dtype: int64
                  - name: type
                    dtype: string
                - name: num_lines
                  struct:
                  - name: default
                    dtype: int64
                  - name: type
                    dtype: string
                - name: pattern
                  struct:
                  - name: type
                    dtype: string
                - name: query
                  struct:
                  - name: type
                    dtype: string
                - name: source
                  struct:
                  - name: type
                    dtype: string
                - name: topn
                  struct:
                  - name: default
                    dtype: int64
                  - name: type
                    dtype: string
                - name: view_source
                  struct:
                  - name: default
                    dtype: bool
                  - name: type
                    dtype: string
              - name: required
                list: string
              - name: type
                dtype: string
      - name: type
        dtype: string
    - name: content_type
      dtype: string
    - name: name
      dtype: string
    - name: recipient
      dtype: string
    - name: role
      dtype: string
  - name: latency_s
    dtype: float64
  - name: error
    dtype: 'null'
  - name: attempts
    dtype: int64
  - name: status
    dtype: string
  - name: chunk_idx
    dtype: int64
  - name: num_chunks
    dtype: int64
  splits:
  - name: train
    num_bytes: 1139271069
    num_examples: 6102
  download_size: 470031205
  dataset_size: 1139271069
- config_name: seed_51
  features:
  - name: qid
    dtype: int64
  - name: question
    dtype: string
  - name: answer
    dtype: string
  - name: messages
    list:
    - name: channel
      dtype: string
    - name: content
      list:
      - name: channel_config
        struct:
        - name: channel_required
          dtype: bool
        - name: valid_channels
          list: string
      - name: conversation_start_date
        dtype: string
      - name: knowledge_cutoff
        dtype: string
      - name: model_identity
        dtype: string
      - name: reasoning_effort
        dtype: string
      - name: text
        dtype: string
      - name: tools
        struct:
        - name: browser
          struct:
          - name: description
            dtype: string
          - name: name
            dtype: string
          - name: tools
            list:
            - name: description
              dtype: string
            - name: name
              dtype: string
            - name: parameters
              struct:
              - name: properties
                struct:
                - name: cursor
                  struct:
                  - name: default
                    dtype: int64
                  - name: type
                    dtype: string
                - name: id
                  struct:
                  - name: default
                    dtype: int64
                  - name: type
                    list: string
                - name: loc
                  struct:
                  - name: default
                    dtype: int64
                  - name: type
                    dtype: string
                - name: num_lines
                  struct:
                  - name: default
                    dtype: int64
                  - name: type
                    dtype: string
                - name: pattern
                  struct:
                  - name: type
                    dtype: string
                - name: query
                  struct:
                  - name: type
                    dtype: string
                - name: source
                  struct:
                  - name: type
                    dtype: string
                - name: topn
                  struct:
                  - name: default
                    dtype: int64
                  - name: type
                    dtype: string
                - name: view_source
                  struct:
                  - name: default
                    dtype: bool
                  - name: type
                    dtype: string
              - name: required
                list: string
              - name: type
                dtype: string
      - name: type
        dtype: string
    - name: content_type
      dtype: string
    - name: name
      dtype: string
    - name: recipient
      dtype: string
    - name: role
      dtype: string
  - name: latency_s
    dtype: float64
  - name: error
    dtype: 'null'
  - name: attempts
    dtype: int64
  - name: status
    dtype: string
  - name: chunk_idx
    dtype: int64
  - name: num_chunks
    dtype: int64
  splits:
  - name: train
    num_bytes: 1154584409
    num_examples: 6102
  download_size: 475934015
  dataset_size: 1154584409
- config_name: seed_52
  features:
  - name: qid
    dtype: int64
  - name: question
    dtype: string
  - name: answer
    dtype: string
  - name: messages
    list:
    - name: channel
      dtype: string
    - name: content
      list:
      - name: channel_config
        struct:
        - name: channel_required
          dtype: bool
        - name: valid_channels
          list: string
      - name: conversation_start_date
        dtype: string
      - name: knowledge_cutoff
        dtype: string
      - name: model_identity
        dtype: string
      - name: reasoning_effort
        dtype: string
      - name: text
        dtype: string
      - name: tools
        struct:
        - name: browser
          struct:
          - name: description
            dtype: string
          - name: name
            dtype: string
          - name: tools
            list:
            - name: description
              dtype: string
            - name: name
              dtype: string
            - name: parameters
              struct:
              - name: properties
                struct:
                - name: cursor
                  struct:
                  - name: default
                    dtype: int64
                  - name: type
                    dtype: string
                - name: id
                  struct:
                  - name: default
                    dtype: int64
                  - name: type
                    list: string
                - name: loc
                  struct:
                  - name: default
                    dtype: int64
                  - name: type
                    dtype: string
                - name: num_lines
                  struct:
                  - name: default
                    dtype: int64
                  - name: type
                    dtype: string
                - name: pattern
                  struct:
                  - name: type
                    dtype: string
                - name: query
                  struct:
                  - name: type
                    dtype: string
                - name: source
                  struct:
                  - name: type
                    dtype: string
                - name: topn
                  struct:
                  - name: default
                    dtype: int64
                  - name: type
                    dtype: string
                - name: view_source
                  struct:
                  - name: default
                    dtype: bool
                  - name: type
                    dtype: string
              - name: required
                list: string
              - name: type
                dtype: string
      - name: type
        dtype: string
    - name: content_type
      dtype: string
    - name: name
      dtype: string
    - name: recipient
      dtype: string
    - name: role
      dtype: string
  - name: latency_s
    dtype: float64
  - name: error
    dtype: string
  - name: attempts
    dtype: int64
  - name: status
    dtype: string
  - name: chunk_idx
    dtype: int64
  - name: num_chunks
    dtype: int64
  splits:
  - name: train
    num_bytes: 1133567180
    num_examples: 6102
  download_size: 467407950
  dataset_size: 1133567180
- config_name: seed_53
  features:
  - name: qid
    dtype: int64
  - name: question
    dtype: string
  - name: answer
    dtype: string
  - name: messages
    list:
    - name: channel
      dtype: string
    - name: content
      list:
      - name: channel_config
        struct:
        - name: channel_required
          dtype: bool
        - name: valid_channels
          list: string
      - name: conversation_start_date
        dtype: string
      - name: knowledge_cutoff
        dtype: string
      - name: model_identity
        dtype: string
      - name: reasoning_effort
        dtype: string
      - name: text
        dtype: string
      - name: tools
        struct:
        - name: browser
          struct:
          - name: description
            dtype: string
          - name: name
            dtype: string
          - name: tools
            list:
            - name: description
              dtype: string
            - name: name
              dtype: string
            - name: parameters
              struct:
              - name: properties
                struct:
                - name: cursor
                  struct:
                  - name: default
                    dtype: int64
                  - name: type
                    dtype: string
                - name: id
                  struct:
                  - name: default
                    dtype: int64
                  - name: type
                    list: string
                - name: loc
                  struct:
                  - name: default
                    dtype: int64
                  - name: type
                    dtype: string
                - name: num_lines
                  struct:
                  - name: default
                    dtype: int64
                  - name: type
                    dtype: string
                - name: pattern
                  struct:
                  - name: type
                    dtype: string
                - name: query
                  struct:
                  - name: type
                    dtype: string
                - name: source
                  struct:
                  - name: type
                    dtype: string
                - name: topn
                  struct:
                  - name: default
                    dtype: int64
                  - name: type
                    dtype: string
                - name: view_source
                  struct:
                  - name: default
                    dtype: bool
                  - name: type
                    dtype: string
              - name: required
                list: string
              - name: type
                dtype: string
      - name: type
        dtype: string
    - name: content_type
      dtype: string
    - name: name
      dtype: string
    - name: recipient
      dtype: string
    - name: role
      dtype: string
  - name: latency_s
    dtype: float64
  - name: error
    dtype: 'null'
  - name: attempts
    dtype: int64
  - name: status
    dtype: string
  - name: chunk_idx
    dtype: int64
  - name: num_chunks
    dtype: int64
  splits:
  - name: train
    num_bytes: 1024473567
    num_examples: 6102
  download_size: 420595363
  dataset_size: 1024473567
- config_name: seed_54
  features:
  - name: qid
    dtype: int64
  - name: question
    dtype: string
  - name: answer
    dtype: string
  - name: messages
    list:
    - name: channel
      dtype: string
    - name: content
      list:
      - name: channel_config
        struct:
        - name: channel_required
          dtype: bool
        - name: valid_channels
          list: string
      - name: conversation_start_date
        dtype: string
      - name: knowledge_cutoff
        dtype: string
      - name: model_identity
        dtype: string
      - name: reasoning_effort
        dtype: string
      - name: text
        dtype: string
      - name: tools
        struct:
        - name: browser
          struct:
          - name: description
            dtype: string
          - name: name
            dtype: string
          - name: tools
            list:
            - name: description
              dtype: string
            - name: name
              dtype: string
            - name: parameters
              struct:
              - name: properties
                struct:
                - name: cursor
                  struct:
                  - name: default
                    dtype: int64
                  - name: type
                    dtype: string
                - name: id
                  struct:
                  - name: default
                    dtype: int64
                  - name: type
                    list: string
                - name: loc
                  struct:
                  - name: default
                    dtype: int64
                  - name: type
                    dtype: string
                - name: num_lines
                  struct:
                  - name: default
                    dtype: int64
                  - name: type
                    dtype: string
                - name: pattern
                  struct:
                  - name: type
                    dtype: string
                - name: query
                  struct:
                  - name: type
                    dtype: string
                - name: source
                  struct:
                  - name: type
                    dtype: string
                - name: topn
                  struct:
                  - name: default
                    dtype: int64
                  - name: type
                    dtype: string
                - name: view_source
                  struct:
                  - name: default
                    dtype: bool
                  - name: type
                    dtype: string
              - name: required
                list: string
              - name: type
                dtype: string
      - name: type
        dtype: string
    - name: content_type
      dtype: string
    - name: name
      dtype: string
    - name: recipient
      dtype: string
    - name: role
      dtype: string
  - name: latency_s
    dtype: float64
  - name: error
    dtype: 'null'
  - name: attempts
    dtype: int64
  - name: status
    dtype: string
  - name: chunk_idx
    dtype: int64
  - name: num_chunks
    dtype: int64
  splits:
  - name: train
    num_bytes: 1122591364
    num_examples: 6102
  download_size: 463334159
  dataset_size: 1122591364
- config_name: seed_55
  features:
  - name: qid
    dtype: int64
  - name: question
    dtype: string
  - name: answer
    dtype: string
  - name: messages
    list:
    - name: channel
      dtype: string
    - name: content
      list:
      - name: channel_config
        struct:
        - name: channel_required
          dtype: bool
        - name: valid_channels
          list: string
      - name: conversation_start_date
        dtype: string
      - name: knowledge_cutoff
        dtype: string
      - name: model_identity
        dtype: string
      - name: reasoning_effort
        dtype: string
      - name: text
        dtype: string
      - name: tools
        struct:
        - name: browser
          struct:
          - name: description
            dtype: string
          - name: name
            dtype: string
          - name: tools
            list:
            - name: description
              dtype: string
            - name: name
              dtype: string
            - name: parameters
              struct:
              - name: properties
                struct:
                - name: cursor
                  struct:
                  - name: default
                    dtype: int64
                  - name: type
                    dtype: string
                - name: id
                  struct:
                  - name: default
                    dtype: int64
                  - name: type
                    list: string
                - name: loc
                  struct:
                  - name: default
                    dtype: int64
                  - name: type
                    dtype: string
                - name: num_lines
                  struct:
                  - name: default
                    dtype: int64
                  - name: type
                    dtype: string
                - name: pattern
                  struct:
                  - name: type
                    dtype: string
                - name: query
                  struct:
                  - name: type
                    dtype: string
                - name: source
                  struct:
                  - name: type
                    dtype: string
                - name: topn
                  struct:
                  - name: default
                    dtype: int64
                  - name: type
                    dtype: string
                - name: view_source
                  struct:
                  - name: default
                    dtype: bool
                  - name: type
                    dtype: string
              - name: required
                list: string
              - name: type
                dtype: string
      - name: type
        dtype: string
    - name: content_type
      dtype: string
    - name: name
      dtype: string
    - name: recipient
      dtype: string
    - name: role
      dtype: string
  - name: latency_s
    dtype: float64
  - name: error
    dtype: 'null'
  - name: attempts
    dtype: int64
  - name: status
    dtype: string
  - name: chunk_idx
    dtype: int64
  - name: num_chunks
    dtype: int64
  splits:
  - name: train
    num_bytes: 1122857864
    num_examples: 6100
  download_size: 462379167
  dataset_size: 1122857864
- config_name: seed_56
  features:
  - name: qid
    dtype: int64
  - name: question
    dtype: string
  - name: answer
    dtype: string
  - name: messages
    list:
    - name: channel
      dtype: string
    - name: content
      list:
      - name: channel_config
        struct:
        - name: channel_required
          dtype: bool
        - name: valid_channels
          list: string
      - name: conversation_start_date
        dtype: string
      - name: knowledge_cutoff
        dtype: string
      - name: model_identity
        dtype: string
      - name: reasoning_effort
        dtype: string
      - name: text
        dtype: string
      - name: tools
        struct:
        - name: browser
          struct:
          - name: description
            dtype: string
          - name: name
            dtype: string
          - name: tools
            list:
            - name: description
              dtype: string
            - name: name
              dtype: string
            - name: parameters
              struct:
              - name: properties
                struct:
                - name: cursor
                  struct:
                  - name: default
                    dtype: int64
                  - name: type
                    dtype: string
                - name: id
                  struct:
                  - name: default
                    dtype: int64
                  - name: type
                    list: string
                - name: loc
                  struct:
                  - name: default
                    dtype: int64
                  - name: type
                    dtype: string
                - name: num_lines
                  struct:
                  - name: default
                    dtype: int64
                  - name: type
                    dtype: string
                - name: pattern
                  struct:
                  - name: type
                    dtype: string
                - name: query
                  struct:
                  - name: type
                    dtype: string
                - name: source
                  struct:
                  - name: type
                    dtype: string
                - name: topn
                  struct:
                  - name: default
                    dtype: int64
                  - name: type
                    dtype: string
                - name: view_source
                  struct:
                  - name: default
                    dtype: bool
                  - name: type
                    dtype: string
              - name: required
                list: string
              - name: type
                dtype: string
      - name: type
        dtype: string
    - name: content_type
      dtype: string
    - name: name
      dtype: string
    - name: recipient
      dtype: string
    - name: role
      dtype: string
  - name: latency_s
    dtype: float64
  - name: error
    dtype: 'null'
  - name: attempts
    dtype: int64
  - name: status
    dtype: string
  - name: chunk_idx
    dtype: int64
  - name: num_chunks
    dtype: int64
  splits:
  - name: train
    num_bytes: 1150224058
    num_examples: 6102
  download_size: 474643458
  dataset_size: 1150224058
- config_name: seed_57
  features:
  - name: qid
    dtype: int64
  - name: question
    dtype: string
  - name: answer
    dtype: string
  - name: messages
    list:
    - name: channel
      dtype: string
    - name: content
      list:
      - name: channel_config
        struct:
        - name: channel_required
          dtype: bool
        - name: valid_channels
          list: string
      - name: conversation_start_date
        dtype: string
      - name: knowledge_cutoff
        dtype: string
      - name: model_identity
        dtype: string
      - name: reasoning_effort
        dtype: string
      - name: text
        dtype: string
      - name: tools
        struct:
        - name: browser
          struct:
          - name: description
            dtype: string
          - name: name
            dtype: string
          - name: tools
            list:
            - name: description
              dtype: string
            - name: name
              dtype: string
            - name: parameters
              struct:
              - name: properties
                struct:
                - name: cursor
                  struct:
                  - name: default
                    dtype: int64
                  - name: type
                    dtype: string
                - name: id
                  struct:
                  - name: default
                    dtype: int64
                  - name: type
                    list: string
                - name: loc
                  struct:
                  - name: default
                    dtype: int64
                  - name: type
                    dtype: string
                - name: num_lines
                  struct:
                  - name: default
                    dtype: int64
                  - name: type
                    dtype: string
                - name: pattern
                  struct:
                  - name: type
                    dtype: string
                - name: query
                  struct:
                  - name: type
                    dtype: string
                - name: source
                  struct:
                  - name: type
                    dtype: string
                - name: topn
                  struct:
                  - name: default
                    dtype: int64
                  - name: type
                    dtype: string
                - name: view_source
                  struct:
                  - name: default
                    dtype: bool
                  - name: type
                    dtype: string
              - name: required
                list: string
              - name: type
                dtype: string
      - name: type
        dtype: string
    - name: content_type
      dtype: string
    - name: name
      dtype: string
    - name: recipient
      dtype: string
    - name: role
      dtype: string
  - name: latency_s
    dtype: float64
  - name: error
    dtype: 'null'
  - name: attempts
    dtype: int64
  - name: status
    dtype: string
  - name: chunk_idx
    dtype: int64
  - name: num_chunks
    dtype: int64
  splits:
  - name: train
    num_bytes: 1131677120
    num_examples: 6102
  download_size: 466987581
  dataset_size: 1131677120
configs:
- config_name: seed_42
  data_files:
  - split: train
    path: seed_42/train-*
- config_name: seed_43
  data_files:
  - split: train
    path: seed_43/train-*
- config_name: seed_44
  data_files:
  - split: train
    path: seed_44/train-*
- config_name: seed_45
  data_files:
  - split: train
    path: seed_45/train-*
- config_name: seed_46
  data_files:
  - split: train
    path: seed_46/train-*
- config_name: seed_47
  data_files:
  - split: train
    path: seed_47/train-*
- config_name: seed_48
  data_files:
  - split: train
    path: seed_48/train-*
- config_name: seed_49
  data_files:
  - split: train
    path: seed_49/train-*
- config_name: seed_50
  data_files:
  - split: train
    path: seed_50/train-*
- config_name: seed_51
  data_files:
  - split: train
    path: seed_51/train-*
- config_name: seed_52
  data_files:
  - split: train
    path: seed_52/train-*
- config_name: seed_53
  data_files:
  - split: train
    path: seed_53/train-*
- config_name: seed_54
  data_files:
  - split: train
    path: seed_54/train-*
- config_name: seed_55
  data_files:
  - split: train
    path: seed_55/train-*
- config_name: seed_56
  data_files:
  - split: train
    path: seed_56/train-*
- config_name: seed_57
  data_files:
  - split: train
    path: seed_57/train-*
license: mit
---
<div style="display: flex; align-items: center; justify-content: center; gap: 8px;">
  <img src="imgs/or-logo1.png" style="height: 84px; width: auto;">
  <img src="imgs/openresearcher-title.svg" style="height: 84px; width: auto;">
</div>


<div align="center">
  <a href="https://x.com/DongfuJiang/status/2020946549422031040"><img src="https://img.shields.io/badge/Twitter-000000?style=for-the-badge&logo=X&logoColor=white" alt="Blog"></a>
  <a href="https://boiled-honeycup-4c7.notion.site/OpenResearcher-A-Fully-Open-Pipeline-for-Long-Horizon-Deep-Research-Trajectory-Synthesis-2f7e290627b5800cb3a0cd7e8d6ec0ea?source=copy_link"><img src="https://img.shields.io/badge/Blog-4285F4?style=for-the-badge&logo=google-chrome&logoColor=white" alt="Blog"></a>
    <a href="https://github.com/TIGER-AI-Lab/OpenResearcher"><img src="https://img.shields.io/badge/Github-181717?style=for-the-badge&logo=github&logoColor=white" alt="Blog"></a>
  <a href="https://huggingface.co/datasets/OpenResearcher/OpenResearcher-Dataset"><img src="https://img.shields.io/badge/Dataset-FFB7B2?style=for-the-badge&logo=huggingface&logoColor=ffffff" alt="Dataset"></a>
  <a href="https://huggingface.co/OpenResearcher/Nemotron-3-Nano-30B-A3B"><img src="https://img.shields.io/badge/Model-FFD966?style=for-the-badge&logo=huggingface&logoColor=ffffff" alt="Model"></a>
  <a href="https://huggingface.co/spaces/OpenResearcher/OpenResearcher"><img src="https://img.shields.io/badge/Demo-F97316.svg?style=for-the-badge&logo=gradio&logoColor=white" alt="Demo"></a>
  <!-- <a href="https://wandb.ai/dongfu/nano-v3-sft-search"><img src="https://img.shields.io/badge/WandB%20Logs-48B5A3?style=for-the-badge&logo=weightsandbiases&logoColor=white" alt="WandB Logs"></a> -->
  <a href="https://huggingface.co/datasets/OpenResearcher/OpenResearcher-Eval-Logs/tree/main"><img src="https://img.shields.io/badge/Eval%20Logs-755BB4?style=for-the-badge&logo=google-sheets&logoColor=white" alt="Eval Logs"></a> 
</div>
</div>
<p align="center">
  🤗 <a href="https://huggingface.co/collections/TIGER-Lab/openresearcher" target="_blank">HuggingFace</a> ｜
<img src="imgs/notion.svg" width="15px" style="display:inline;"> <a href="https://boiled-honeycup-4c7.notion.site/OpenResearcher-A-Fully-Open-Pipeline-for-Long-Horizon-Deep-Research-Trajectory-Synthesis-2f7e290627b5800cb3a0cd7e8d6ec0ea?source=copy_link" target="_blank">Blog</a> ｜ <img src="imgs/slack.png" width="14px" style="display:inline;"> <a href="https://join.slack.com/t/openresearcher/shared_invite/zt-3p0r32cky-PqtZkVjjWIAI14~XwcRMfQ" target="_blank">Slack</a> | <img src="imgs/wechat.svg" width="14px" style="display:inline;"> <a href="https://github.com/TIGER-AI-Lab/OpenResearcher/blob/main/assets/imgs/wechat_group.jpg" target="_blank">WeChat</a> 

</p>

## Overview 
**OpenResearcher** is a fully open agentic large language model (30B-A3B) designed for **long-horizon deep research** scenarios. It achieves an impressive **54.8%** accuracy on [BrowseComp-Plus](https://huggingface.co/spaces/Tevatron/BrowseComp-Plus), surpassing performance of `GPT-4.1`, `Claude-Opus-4`, `Gemini-2.5-Pro`, `DeepSeek-R1` and `Tongyi-DeepResearch`. It also demonstrates **leading performance** across a range of deep research benchmarks, including BrowseComp, GAIA, WebWalkerQA, and xbench-DeepSearch. We **fully open-source** the training and evaluation recipe—including data, model, training methodology, and evaluation framework for everyone to progress deep research.
## OpenResearcher Training Dataset

Our training dataset consists of **96K** high-quality long-horizon DeepResearch trajectories with **100+ turns** generated by GPT-OSS-120B using its [native browser tools](https://docs.vllm.ai/projects/recipes/en/latest/OpenAI/GPT-OSS.html#usage:~:text=Limitation%20section%20below.-,Tool%20Use,-%C2%B6). To enable scalable and cost-efficient data generation, we deploy a self-hosted search engine over carefully constructed ~11B-token [corpus](https://huggingface.co/datasets/OpenResearcher/OpenResearcher-Corpus)
, completely eliminating reliance on external search APIs.


## Format
Each row in the dataset contains the following fields:

- **qid (int64)**: A unique identifier for each question or task.

- **question (string)**: The original deepresearch question compiled from [MiroVerse](https://huggingface.co/datasets/miromind-ai/MiroVerse-v0.1).

- **answer (string)**: The final answer to the question.

- **messages (list)**: A list of messages representing the GPT-OSS 120B deep research trajectory, including intermediate reasoning steps, tool calls, observations, and model responses throughout the problem-solving process.

## Citation

```bibtex
@misc{li2025openresearcher,
  title={OpenResearcher: A Fully Open Pipeline for Long-Horizon Deep Research Trajectory Synthesis},
  author={Zhuofeng Li and Dongfu Jiang and Xueguang Ma and Haoxiang Zhang and Ping Nie and Yuyu Zhang and Kai Zou and Jianwen Xie and Yu Zhang and Wenhu Chen},
  year={2025},
  howpublished={\url{https://www.notion.so/OpenResearcher-A-Fully-Open-Pipeline-for-Long-Horizon-Deep-Research-Trajectory-Synthesis-2f7e290627b5800cb3a0cd7e8d6ec0ea}},
  note={Notion Blog}
}
```