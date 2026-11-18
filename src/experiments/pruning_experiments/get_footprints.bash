{
  echo "h,training,deployment"
  for h in h000 h001 h002 h003 h004 h005 h006 h007 h008 h010 h012 h015; do
    train_count=$(wc -c < "${h}/results/stage_21_overall_keys_epsilon_1.0.bin")
    deploy_count=$(wc -c < "${h}/results/stage_21_overall_policykeys_epsilon_1.0.bin")
    echo "$((10#${h#h})),$((train_count / 8)),$((deploy_count / 8))"
  done
} > footprint.csv
