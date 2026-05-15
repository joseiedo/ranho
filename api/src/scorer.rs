use crate::types::Label;

const FRAUD_THRESHOLD: f32 = 0.6;

pub fn score(neighbors: [Label; 5]) -> (f32, bool) {
    let fraud_count = neighbors.iter().filter(|&&l| l == Label::Fraud).count();
    let fraud_score = fraud_count as f32 / 5.0;
    let approved = fraud_score < FRAUD_THRESHOLD;
    (fraud_score, approved)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_legit() {
        let (score, approved) = score([Label::Legit; 5]);
        assert_eq!(score, 0.0);
        assert!(approved);
    }

    #[test]
    fn all_fraud() {
        let (score, approved) = score([Label::Fraud; 5]);
        assert_eq!(score, 1.0);
        assert!(!approved);
    }

    #[test]
    fn three_fraud_is_denied() {
        let neighbors = [Label::Fraud, Label::Fraud, Label::Fraud, Label::Legit, Label::Legit];
        let (score, approved) = score(neighbors);
        assert_eq!(score, 0.6);
        assert!(!approved);
    }

    #[test]
    fn two_fraud_is_approved() {
        let neighbors = [Label::Fraud, Label::Fraud, Label::Legit, Label::Legit, Label::Legit];
        let (score, approved) = score(neighbors);
        assert_eq!(score, 0.4);
        assert!(approved);
    }
}
