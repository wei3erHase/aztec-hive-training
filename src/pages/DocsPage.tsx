import React from 'react';
import { motion } from 'motion/react';
import {
  features,
  architectureLayers,
  workflow,
  eli5Architectures,
  techSpecs,
} from '../config/docsContent';
import { useAppNavigation } from '../providers/AppNavigationContext';

const styles = {
  page: 'mx-auto w-full max-w-7xl px-4 py-12 sm:px-6 sm:py-16',
  section: 'mb-24',
  sectionHeader:
    'font-display text-xl font-bold text-default mb-6 text-center sm:text-2xl sm:mb-8',
  sectionSubtitle:
    'mx-auto mb-8 max-w-xl text-center text-base text-muted sm:mb-10 sm:text-sm',
  badge:
    'flex w-fit mx-auto items-center gap-2 px-4 py-2 bg-hive-600/20 border border-hive-500/30 rounded-full text-hive-300 text-sm mb-6',
  badgeDot: 'w-2 h-2 bg-hive-400 rounded-full',
  heroTitle:
    'relative z-10 inline-block pb-1 font-display text-3xl font-bold leading-[1.08] mb-6 bg-gradient-to-r from-[var(--default)] via-[var(--default)] to-[var(--accent-primary)] bg-clip-text text-transparent sm:text-4xl md:text-6xl',
  heroSubtitle:
    'mx-auto max-w-3xl text-base text-muted leading-relaxed sm:text-lg md:text-xl',
  featureGrid: 'grid gap-4 sm:gap-6 md:grid-cols-2',
  featureCard:
    'group p-4 bg-surface-secondary border border-default rounded-2xl transition-all duration-300 hover:border-[var(--accent-primary)]/40 sm:p-6',
  featureIconWrapper:
    'w-14 h-14 bg-gradient-to-br from-neural-600/30 to-hive-600/30 rounded-xl flex items-center justify-center text-3xl shrink-0 group-hover:scale-110 transition-transform',
  featureTitle: 'mb-2 text-base font-semibold text-default sm:text-lg',
  featureDesc: 'text-base leading-relaxed text-muted sm:text-sm',
  workflowGrid: 'grid grid-cols-2 gap-4 sm:gap-6 md:grid-cols-4',
  workflowBox:
    'relative z-10 mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-2xl border border-default bg-surface-secondary text-3xl sm:h-20 sm:w-20 sm:text-4xl',
  workflowStep:
    'absolute -top-2 -right-2 w-7 h-7 bg-accent rounded-lg flex items-center justify-center text-xs font-bold text-on-accent',
  workflowConnector:
    'hidden md:block absolute top-1/2 left-0 right-0 h-0.5 bg-gradient-to-r from-transparent via-neural-500/30 to-transparent -translate-y-1/2',
  archStack: 'max-w-2xl mx-auto space-y-4',
  archLayer: 'rounded-xl p-4 bg-gradient-to-r sm:p-5',
  archLabel: 'text-base font-semibold text-default sm:text-lg',
  archChip:
    'rounded-lg bg-interactive px-3 py-1 text-base text-default backdrop-blur sm:text-sm',
  archConnector: 'flex justify-center my-2',
  specsGrid: 'grid gap-4 sm:gap-6 md:grid-cols-3',
  specCard: 'rounded-2xl border border-default bg-surface-secondary p-4 sm:p-6',
  specIconWrapper:
    'w-10 h-10 rounded-xl flex items-center justify-center text-xl mb-4',
  specTitle: 'mb-3 text-base font-semibold text-default sm:text-lg',
  specList: 'space-y-2 text-base text-muted sm:text-sm',
  specDot: 'w-1.5 h-1.5 rounded-full',
  eli5Wrapper: 'mt-12',
  eli5Title: 'mb-4 text-lg font-semibold text-default',
  eli5Card: 'rounded-xl border border-default bg-surface-secondary p-4',
  eli5Name: 'font-medium text-neural-300',
  eli5Desc: 'mt-1 text-sm text-muted',
  aztecNote: 'mt-12 rounded-xl border border-aztec-jade/20 bg-aztec-jade/5 p-4',
  aztecTitle: 'mb-3 text-lg font-semibold text-aztec-turquoise',
  aztecBody: 'mb-3 text-sm text-muted',
  aztecList: 'space-y-2 text-sm text-muted',
  aztecHighlight: 'font-medium text-aztec-jade',
  ctaBox:
    'rounded-2xl border border-[var(--accent-primary)]/20 bg-surface-secondary p-6 sm:rounded-3xl sm:p-10',
  ctaTitle: 'font-display text-xl font-bold text-default mb-4 sm:text-2xl',
  ctaSubtitle: 'mx-auto mb-6 max-w-md text-base text-muted sm:text-sm',
  ctaRow: 'flex flex-wrap items-center justify-center gap-4',
  ctaPrimary:
    'px-6 py-3 gradient-primary font-semibold rounded-xl shadow-theme hover:shadow-theme-hover transition-all',
  ctaSecondary:
    'px-6 py-3 border border-default text-muted font-semibold rounded-xl hover:border-[var(--accent-primary)]/40 hover:text-default transition-colors',
} as const;

export function DocsPage() {
  const { navigate } = useAppNavigation();

  return (
    <div className={styles.page} data-testid="docs-page">
      {/* Header */}
      <motion.section
        className="relative z-10 text-center mb-20"
        initial={{ opacity: 0, y: 30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
      >
        <h1 className={styles.heroTitle}>What is Hive Training?</h1>
        <motion.div
          className={styles.badge}
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.2 }}
        >
          <span className={styles.badgeDot} />
          Documentation
        </motion.div>
        <p className={styles.heroSubtitle}>
          HIVE stands for Hidden Inference Verification Engine. It is
          privacy-preserving digit recognition on Aztec: your drawings never
          leave your device, only ZK proofs go on-chain. Model weights are
          public; train together, verify together.
        </p>
      </motion.section>

      {/* Features */}
      <motion.section
        className={styles.section}
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.3 }}
      >
        <h2 className={styles.sectionHeader}>Core Features</h2>
        <div className={styles.featureGrid}>
          {features.map((feature, i) => (
            <motion.div
              key={feature.title}
              className={styles.featureCard}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.4 + i * 0.1 }}
              whileHover={{ y: -4, transition: { duration: 0.2 } }}
            >
              <div className="flex items-start gap-4">
                <div className={styles.featureIconWrapper}>{feature.icon}</div>
                <div>
                  <h3 className={styles.featureTitle}>{feature.title}</h3>
                  <p className={styles.featureDesc}>{feature.description}</p>
                </div>
              </div>
            </motion.div>
          ))}
        </div>
      </motion.section>

      {/* How It Works */}
      <motion.section
        className={styles.section}
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.5 }}
      >
        <h2 className={styles.sectionHeader}>How It Works</h2>
        <p className={styles.sectionSubtitle}>
          From drawing to verified prediction in four steps
        </p>
        <div className="relative">
          <div className={styles.workflowConnector} />
          <div className={styles.workflowGrid}>
            {workflow.map((item, i) => (
              <motion.div
                key={item.step}
                className="relative text-center"
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: 0.6 + i * 0.15 }}
              >
                <motion.div
                  className={styles.workflowBox}
                  whileHover={{ scale: 1.1, rotate: 5 }}
                >
                  {item.icon}
                  <div className={styles.workflowStep}>{item.step}</div>
                </motion.div>
                <h4 className="mb-1 text-sm font-semibold text-default sm:text-base">
                  {item.title}
                </h4>
                <p className="text-base text-muted sm:text-sm">
                  {item.description}
                </p>
              </motion.div>
            ))}
          </div>
        </div>
      </motion.section>

      {/* Architecture */}
      <motion.section
        className={styles.section}
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.6 }}
      >
        <h2 className={styles.sectionHeader}>Architecture</h2>
        <p className={styles.sectionSubtitle}>
          Aztec-first: Noir circuits, Aztec Network, ZK proofs
        </p>
        <div className={styles.archStack}>
          {architectureLayers.map((layer, i) => (
            <motion.div
              key={layer.label}
              initial={{ opacity: 0, x: -30 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.7 + i * 0.1 }}
            >
              <div className={`${styles.archLayer} ${layer.color}`}>
                <div className="flex items-center justify-between flex-wrap gap-4">
                  <span className={styles.archLabel}>{layer.label}</span>
                  <div className="flex flex-wrap gap-2">
                    {layer.items.map((item) => (
                      <span key={item} className={styles.archChip}>
                        {item}
                      </span>
                    ))}
                  </div>
                </div>
              </div>
              {i < architectureLayers.length - 1 && (
                <div className={styles.archConnector}>
                  <div className="w-0.5 h-6 bg-[var(--border-color)]" />
                </div>
              )}
            </motion.div>
          ))}
        </div>
      </motion.section>

      {/* Tech Specs */}
      <motion.section
        className={styles.section}
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.7 }}
      >
        <h2 className={styles.sectionHeader}>Technical Specifications</h2>
        <div className={styles.specsGrid}>
          {techSpecs.map((spec, i) => (
            <motion.div
              key={spec.title}
              className={styles.specCard}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.8 + i * 0.05 }}
            >
              <div className={`${styles.specIconWrapper} ${spec.iconBg}`}>
                {spec.icon}
              </div>
              <h3 className={styles.specTitle}>{spec.title}</h3>
              <ul className={styles.specList}>
                {spec.items.map((item) => (
                  <li key={item} className="flex items-center gap-2">
                    <span className={`${styles.specDot} ${spec.dotColor}`} />
                    {item}
                  </li>
                ))}
              </ul>
            </motion.div>
          ))}
        </div>

        <div className={styles.eli5Wrapper}>
          <h3 className={styles.eli5Title}>Architectures (plain English)</h3>
          <div className="space-y-3">
            {eli5Architectures.map((arch) => (
              <div key={arch.name} className={styles.eli5Card}>
                <span className={styles.eli5Name}>{arch.name}</span>
                <p className={styles.eli5Desc}>{arch.eli5}</p>
              </div>
            ))}
          </div>
        </div>

        <div className={styles.aztecNote}>
          <h3 className={styles.aztecTitle}>Working within Aztec</h3>
          <p className={styles.aztecBody}>
            Hive is designed for Aztec&apos;s privacy-first VM. These parameters
            shape how we pack weights and run explainability:
          </p>
          <ul className={styles.aztecList}>
            <li>
              <span className={styles.aztecHighlight}>
                Up to 64 public writes per tx
              </span>{' '}
              — We pack weights into Fields to fit. SingleLayer ~60, MLP 44, CNN
              4.
            </li>
            <li>
              <span className={styles.aztecHighlight}>6M L2 gas per tx</span> —
              Shapley for CNN uses most of this on devnet. MLP Explain fits
              comfortably; Predict and Train work with all architectures.
            </li>
          </ul>
        </div>
      </motion.section>

      {/* CTA */}
      <motion.section
        className="text-center"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.9 }}
      >
        <div className={styles.ctaBox}>
          <h2 className={styles.ctaTitle}>Ready to Try It?</h2>
          <p className={styles.ctaSubtitle}>
            Connect your wallet and start training the neural network with your
            own handwritten digits.
          </p>
          <div className={styles.ctaRow}>
            <motion.button
              onClick={() => navigate('home')}
              className={styles.ctaPrimary}
              whileHover={{
                scale: 1.02,
                boxShadow: '0 0 30px rgba(168, 85, 247, 0.4)',
              }}
              whileTap={{ scale: 0.98 }}
            >
              Try Hive Training
            </motion.button>
            <motion.a
              href="https://github.com/wei3erhase/aztec-hive-training"
              target="_blank"
              rel="noreferrer"
              className={styles.ctaSecondary}
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
            >
              View on GitHub
            </motion.a>
          </div>
        </div>
      </motion.section>
    </div>
  );
}
