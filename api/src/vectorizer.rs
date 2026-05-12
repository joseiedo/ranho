use chrono::{Datelike, Timelike};
use std::collections::HashMap;

use crate::types::TransactionPayload;

const MAX_AMOUNT: f32 = 10_000.0;
const MAX_INSTALLMENTS: f32 = 12.0;
const AMOUNT_VS_AVG_RATIO: f32 = 10.0;
const MAX_MINUTES: f32 = 1440.0;
const MAX_KM: f32 = 1000.0;
const MAX_TX_COUNT_24H: f32 = 20.0;
const MAX_MERCHANT_AVG_AMOUNT: f32 = 10_000.0;
const DEFAULT_MCC_RISK: f32 = 0.5;

const HOUR_LUT: [f32; 24] = [
    0.0 / 23.0,  1.0 / 23.0,  2.0 / 23.0,  3.0 / 23.0,
    4.0 / 23.0,  5.0 / 23.0,  6.0 / 23.0,  7.0 / 23.0,
    8.0 / 23.0,  9.0 / 23.0, 10.0 / 23.0, 11.0 / 23.0,
   12.0 / 23.0, 13.0 / 23.0, 14.0 / 23.0, 15.0 / 23.0,
   16.0 / 23.0, 17.0 / 23.0, 18.0 / 23.0, 19.0 / 23.0,
   20.0 / 23.0, 21.0 / 23.0, 22.0 / 23.0, 23.0 / 23.0,
];

const DOW_LUT: [f32; 7] = [
    0.0 / 6.0, 1.0 / 6.0, 2.0 / 6.0, 3.0 / 6.0,
    4.0 / 6.0, 5.0 / 6.0, 6.0 / 6.0,
];

fn clamp01(x: f32) -> f32 {
    x.clamp(0.0, 1.0)
}

pub struct Vectorizer {
    mcc_risk: HashMap<String, f32>,
}

impl Vectorizer {
    pub fn new(mcc_risk: HashMap<String, f32>) -> Self {
        Self { mcc_risk }
    }

    pub fn vectorize(&self, payload: &TransactionPayload) -> [f32; 14] {
        let tx = &payload.transaction;
        let customer = &payload.customer;
        let merchant = &payload.merchant;
        let terminal = &payload.terminal;

        let hour = tx.requested_at.hour() as usize;
        let dow = tx.requested_at.weekday().num_days_from_monday() as usize;

        let (minutes_since_last, km_from_last) = match &payload.last_transaction {
            Some(last) => {
                let diff_minutes = (tx.requested_at - last.timestamp).num_minutes() as f32;
                (
                    clamp01(diff_minutes / MAX_MINUTES),
                    clamp01(last.km_from_current / MAX_KM),
                )
            }
            None => (-1.0, -1.0),
        };

        let unknown_merchant = if customer.known_merchants.iter().any(|m| m == &merchant.id) {
            0.0
        } else {
            1.0
        };

        let mcc_risk = *self.mcc_risk.get(&merchant.mcc).unwrap_or(&DEFAULT_MCC_RISK);

        [
            clamp01(tx.amount / MAX_AMOUNT),                                  // 0  amount
            clamp01(tx.installments as f32 / MAX_INSTALLMENTS),               // 1  installments
            clamp01((tx.amount / customer.avg_amount) / AMOUNT_VS_AVG_RATIO), // 2  amount_vs_avg
            HOUR_LUT[hour],                                                    // 3  hour_of_day
            DOW_LUT[dow],                                                      // 4  day_of_week
            minutes_since_last,                                                // 5  minutes_since_last_tx
            km_from_last,                                                      // 6  km_from_last_tx
            clamp01(terminal.km_from_home / MAX_KM),                          // 7  km_from_home
            clamp01(customer.tx_count_24h as f32 / MAX_TX_COUNT_24H),         // 8  tx_count_24h
            if terminal.is_online { 1.0 } else { 0.0 },                       // 9  is_online
            if terminal.card_present { 1.0 } else { 0.0 },                    // 10 card_present
            unknown_merchant,                                                   // 11 unknown_merchant
            mcc_risk,                                                           // 12 mcc_risk
            clamp01(merchant.avg_amount / MAX_MERCHANT_AVG_AMOUNT),            // 13 merchant_avg_amount
        ]
    }

    /// Quantize a float vector to i8.
    /// Range [0.0, 1.0] maps to [0, 127].
    /// Sentinel -1.0 maps to -127 naturally: (-1.0 * 127.0).round() = -127.
    pub fn quantize(vector: &[f32; 14]) -> [i8; 14] {
        let mut out = [0i8; 14];
        for (i, &v) in vector.iter().enumerate() {
            out[i] = (v * 127.0).round().clamp(-127.0, 127.0) as i8;
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Customer, LastTransaction, Merchant, Terminal, Transaction, TransactionPayload};

    fn mcc_map(entries: &[(&str, f32)]) -> HashMap<String, f32> {
        entries.iter().map(|(k, v)| (k.to_string(), *v)).collect()
    }

    fn base_payload() -> TransactionPayload {
        TransactionPayload {
            id: "test".into(),
            transaction: Transaction {
                amount: 500.0,
                installments: 1,
                requested_at: "2026-01-05T12:00:00Z".parse().unwrap(), // Monday noon
            },
            customer: Customer {
                avg_amount: 1000.0,
                tx_count_24h: 2,
                known_merchants: vec!["MERC-001".to_string()],
            },
            merchant: Merchant {
                id: "MERC-001".into(),
                mcc: "5912".into(),
                avg_amount: 300.0,
            },
            terminal: Terminal {
                is_online: false,
                card_present: true,
                km_from_home: 10.0,
            },
            last_transaction: None,
        }
    }

    fn vec_for(payload: &TransactionPayload) -> [f32; 14] {
        Vectorizer::new(mcc_map(&[("5912", 0.20)])).vectorize(payload)
    }

    // ── dim 0: amount ────────────────────────────────────────────────────────

    #[test]
    fn dim0_normal() {
        let mut p = base_payload();
        p.transaction.amount = 5000.0;
        assert!((vec_for(&p)[0] - 0.5).abs() < 1e-5);
    }

    #[test]
    fn dim0_clamped_above_max() {
        let mut p = base_payload();
        p.transaction.amount = 15_000.0;
        assert_eq!(vec_for(&p)[0], 1.0);
    }

    #[test]
    fn dim0_zero() {
        let mut p = base_payload();
        p.transaction.amount = 0.0;
        assert_eq!(vec_for(&p)[0], 0.0);
    }

    // ── dim 1: installments ──────────────────────────────────────────────────

    #[test]
    fn dim1_normal() {
        let mut p = base_payload();
        p.transaction.installments = 6;
        assert!((vec_for(&p)[1] - 0.5).abs() < 1e-5);
    }

    #[test]
    fn dim1_clamped_above_max() {
        let mut p = base_payload();
        p.transaction.installments = 24;
        assert_eq!(vec_for(&p)[1], 1.0);
    }

    #[test]
    fn dim1_zero() {
        let mut p = base_payload();
        p.transaction.installments = 0;
        assert_eq!(vec_for(&p)[1], 0.0);
    }

    // ── dim 2: amount_vs_avg ─────────────────────────────────────────────────

    #[test]
    fn dim2_normal() {
        // amount=500, avg=1000 → (0.5) / 10 = 0.05
        let mut p = base_payload();
        p.transaction.amount = 500.0;
        p.customer.avg_amount = 1000.0;
        assert!((vec_for(&p)[2] - 0.05).abs() < 1e-5);
    }

    #[test]
    fn dim2_clamped_above_max() {
        // amount=10000, avg=100 → ratio=100 → /10 = 10 → clamped 1.0
        let mut p = base_payload();
        p.transaction.amount = 10_000.0;
        p.customer.avg_amount = 100.0;
        assert_eq!(vec_for(&p)[2], 1.0);
    }

    // ── dim 3: hour_of_day ───────────────────────────────────────────────────

    #[test]
    fn dim3_midnight() {
        let mut p = base_payload();
        p.transaction.requested_at = "2026-01-05T00:00:00Z".parse().unwrap();
        assert_eq!(vec_for(&p)[3], 0.0);
    }

    #[test]
    fn dim3_last_hour() {
        let mut p = base_payload();
        p.transaction.requested_at = "2026-01-05T23:00:00Z".parse().unwrap();
        assert_eq!(vec_for(&p)[3], 1.0);
    }

    #[test]
    fn dim3_noon() {
        let mut p = base_payload();
        p.transaction.requested_at = "2026-01-05T12:00:00Z".parse().unwrap();
        assert!((vec_for(&p)[3] - 12.0 / 23.0).abs() < 1e-5);
    }

    // ── dim 4: day_of_week ───────────────────────────────────────────────────

    #[test]
    fn dim4_monday() {
        // 2026-01-05 is Monday (Jan 1=Thu → +4 = Mon)
        let mut p = base_payload();
        p.transaction.requested_at = "2026-01-05T12:00:00Z".parse().unwrap();
        assert_eq!(vec_for(&p)[4], 0.0);
    }

    #[test]
    fn dim4_sunday() {
        // 2026-01-11 is Sunday
        let mut p = base_payload();
        p.transaction.requested_at = "2026-01-11T12:00:00Z".parse().unwrap();
        assert_eq!(vec_for(&p)[4], 1.0);
    }

    // ── dim 5: minutes_since_last_tx ─────────────────────────────────────────

    #[test]
    fn dim5_sentinel_when_no_last_tx() {
        let p = base_payload();
        assert_eq!(vec_for(&p)[5], -1.0);
    }

    #[test]
    fn dim5_normal() {
        // 720 minutes ago → 720/1440 = 0.5
        let mut p = base_payload();
        p.transaction.requested_at = "2026-01-05T12:00:00Z".parse().unwrap();
        p.last_transaction = Some(LastTransaction {
            timestamp: "2026-01-05T00:00:00Z".parse().unwrap(),
            km_from_current: 0.0,
        });
        assert!((vec_for(&p)[5] - 0.5).abs() < 1e-5);
    }

    #[test]
    fn dim5_clamped_above_max() {
        let mut p = base_payload();
        p.transaction.requested_at = "2026-01-06T12:00:00Z".parse().unwrap();
        p.last_transaction = Some(LastTransaction {
            timestamp: "2026-01-05T00:00:00Z".parse().unwrap(), // 36h ago
            km_from_current: 0.0,
        });
        assert_eq!(vec_for(&p)[5], 1.0);
    }

    // ── dim 6: km_from_last_tx ───────────────────────────────────────────────

    #[test]
    fn dim6_sentinel_when_no_last_tx() {
        let p = base_payload();
        assert_eq!(vec_for(&p)[6], -1.0);
    }

    #[test]
    fn dim6_normal() {
        let mut p = base_payload();
        p.last_transaction = Some(LastTransaction {
            timestamp: "2026-01-05T06:00:00Z".parse().unwrap(),
            km_from_current: 500.0,
        });
        assert!((vec_for(&p)[6] - 0.5).abs() < 1e-5);
    }

    #[test]
    fn dim6_clamped_above_max() {
        let mut p = base_payload();
        p.last_transaction = Some(LastTransaction {
            timestamp: "2026-01-05T06:00:00Z".parse().unwrap(),
            km_from_current: 2000.0,
        });
        assert_eq!(vec_for(&p)[6], 1.0);
    }

    // ── dim 7: km_from_home ──────────────────────────────────────────────────

    #[test]
    fn dim7_normal() {
        let mut p = base_payload();
        p.terminal.km_from_home = 500.0;
        assert!((vec_for(&p)[7] - 0.5).abs() < 1e-5);
    }

    #[test]
    fn dim7_clamped() {
        let mut p = base_payload();
        p.terminal.km_from_home = 2000.0;
        assert_eq!(vec_for(&p)[7], 1.0);
    }

    #[test]
    fn dim7_zero() {
        let mut p = base_payload();
        p.terminal.km_from_home = 0.0;
        assert_eq!(vec_for(&p)[7], 0.0);
    }

    // ── dim 8: tx_count_24h ──────────────────────────────────────────────────

    #[test]
    fn dim8_normal() {
        let mut p = base_payload();
        p.customer.tx_count_24h = 10;
        assert!((vec_for(&p)[8] - 0.5).abs() < 1e-5);
    }

    #[test]
    fn dim8_clamped() {
        let mut p = base_payload();
        p.customer.tx_count_24h = 40;
        assert_eq!(vec_for(&p)[8], 1.0);
    }

    // ── dim 9: is_online ─────────────────────────────────────────────────────

    #[test]
    fn dim9_online() {
        let mut p = base_payload();
        p.terminal.is_online = true;
        assert_eq!(vec_for(&p)[9], 1.0);
    }

    #[test]
    fn dim9_in_person() {
        let mut p = base_payload();
        p.terminal.is_online = false;
        assert_eq!(vec_for(&p)[9], 0.0);
    }

    // ── dim 10: card_present ─────────────────────────────────────────────────

    #[test]
    fn dim10_present() {
        let mut p = base_payload();
        p.terminal.card_present = true;
        assert_eq!(vec_for(&p)[10], 1.0);
    }

    #[test]
    fn dim10_absent() {
        let mut p = base_payload();
        p.terminal.card_present = false;
        assert_eq!(vec_for(&p)[10], 0.0);
    }

    // ── dim 11: unknown_merchant ─────────────────────────────────────────────

    #[test]
    fn dim11_known_merchant() {
        let mut p = base_payload();
        p.merchant.id = "MERC-001".into();
        p.customer.known_merchants = vec!["MERC-001".to_string()];
        assert_eq!(vec_for(&p)[11], 0.0);
    }

    #[test]
    fn dim11_unknown_merchant() {
        let mut p = base_payload();
        p.merchant.id = "MERC-999".into();
        p.customer.known_merchants = vec!["MERC-001".to_string()];
        assert_eq!(vec_for(&p)[11], 1.0);
    }

    // ── dim 12: mcc_risk ─────────────────────────────────────────────────────

    #[test]
    fn dim12_known_mcc() {
        let mut p = base_payload();
        p.merchant.mcc = "5912".into();
        assert!((vec_for(&p)[12] - 0.20).abs() < 1e-5);
    }

    #[test]
    fn dim12_unknown_mcc_defaults_to_0_5() {
        let mut p = base_payload();
        p.merchant.mcc = "9999".into();
        assert!((vec_for(&p)[12] - 0.5).abs() < 1e-5);
    }

    // ── dim 13: merchant_avg_amount ──────────────────────────────────────────

    #[test]
    fn dim13_normal() {
        let mut p = base_payload();
        p.merchant.avg_amount = 5000.0;
        assert!((vec_for(&p)[13] - 0.5).abs() < 1e-5);
    }

    #[test]
    fn dim13_clamped() {
        let mut p = base_payload();
        p.merchant.avg_amount = 20_000.0;
        assert_eq!(vec_for(&p)[13], 1.0);
    }

    // ── quantize ─────────────────────────────────────────────────────────────

    #[test]
    fn quantize_zero() {
        let v = [0.0f32; 14];
        assert_eq!(Vectorizer::quantize(&v), [0i8; 14]);
    }

    #[test]
    fn quantize_one() {
        let v = [1.0f32; 14];
        assert_eq!(Vectorizer::quantize(&v), [127i8; 14]);
    }

    #[test]
    fn quantize_sentinel() {
        let mut v = [0.0f32; 14];
        v[5] = -1.0;
        v[6] = -1.0;
        let q = Vectorizer::quantize(&v);
        assert_eq!(q[5], -127);
        assert_eq!(q[6], -127);
    }
}
