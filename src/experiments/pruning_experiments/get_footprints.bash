{
  echo "Policy,Training,Deployment"
  for h in h000 h001 h002 h005 h010; do
    train_count=$(wc -l < "training_policy${h}.csv")
    deploy_count=$(wc -l < "deployment_policy${h}.csv")
    echo "${h},${train_count},${deploy_count}"
  done
} > policy_counts.csv