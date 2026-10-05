"""
campy-benchmarks / locomo / dataset.py
LoCoMo Benchmark Dataset: Multi-Session Conversations & Dynamic Constraint Updates.
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class ProbeQuestion:
    """Judged by scoring.judge (scorer v2) -- see scoring.py for the rules."""
    id: str
    question: str
    expected: str
    kind: str = "value"  # "value" | "negative"
    accept: List[str] = field(default_factory=list)
    stale: List[str] = field(default_factory=list)
    is_deprecation: bool = False


@dataclass
class LoCoMoScenario:
    id: str
    title: str
    sessions: List[List[Dict[str, str]]]  # List of turns per session
    probes: List[ProbeQuestion]


def get_locomo_scenarios(smoke: bool = False) -> List[LoCoMoScenario]:
    """Return scenarios for LoCoMo evaluation.
    
    If smoke=True, returns a focused 3-scenario subset.
    Otherwise, returns full 25 multi-session scenarios.
    """
    scenarios: List[LoCoMoScenario] = []

    # Scenario 1: Database Migration & Schema Constraint
    s1 = LoCoMoScenario(
        id="locomo_01_db_migration",
        title="Database Migration & Schema Constraint",
        sessions=[
            # Session 1
            [
                {"role": "user", "content": "Our service uses PostgreSQL 14 hosted on AWS RDS. Constraint: all tables must use UUID primary keys."},
                {"role": "assistant", "content": "Noted. Primary database is PostgreSQL 14 with UUID primary keys across all tables."},
            ],
            # Session 2
            [
                {"role": "user", "content": "We noticed high index bloat with random UUIDs. Update constraint: new analytics tables should use BIGINT sequence IDs."},
                {"role": "assistant", "content": "Understood. Updated constraint: analytics tables use BIGINT sequence IDs to prevent index bloat."},
            ],
            # Session 3
            [
                {"role": "user", "content": "CRITICAL UPDATE: We have completely migrated from PostgreSQL 14 to PostgreSQL 16. Constraint: do NOT use PostgreSQL 14 anymore, all new deployments target PostgreSQL 16."},
                {"role": "assistant", "content": "Acknowledged. PostgreSQL 14 is deprecated. All deployments must strictly target PostgreSQL 16."},
            ],
        ],
        probes=[
            ProbeQuestion(
                id="p1_current_db",
                question="What is our active production database engine and version?",
                expected="PostgreSQL 16",
                accept=[r"Postgre(?:SQL|s)?\s*16\b"],
                stale=[r"Postgre(?:SQL|s)?\s*14\b"],
                is_deprecation=True,
            ),
            ProbeQuestion(
                id="p2_deprecated_check",
                question="Can we deploy the new payment service on PostgreSQL 14?",
                expected="No, PostgreSQL 14 is deprecated; use PostgreSQL 16.",
                kind="negative",
                accept=[r"Postgre(?:SQL|s)?\s*16\b", r"\bdeprecat\w*", r"\bno\s+longer\b", r"\bmigrat\w*"],
                is_deprecation=True,
            ),
            ProbeQuestion(
                id="p3_analytics_pk",
                question="What primary key format is required for analytics tables?",
                # The update scopes BIGINT to *new* analytics tables; the
                # earlier rule (UUID for all tables) still covers the rest. A
                # gold of bare "BIGINT sequence IDs" made the judge fail the
                # answer that states both (2026-10-05, R1 on hippocampy#278).
                expected=("BIGINT sequence IDs for new analytics tables "
                          "(the earlier rule for all tables is UUID primary keys)"),
                accept=[r"\bBIGINT\b"],
                is_deprecation=False,
            ),
        ],
    )
    scenarios.append(s1)

    # Scenario 2: Authentication Protocol Evolution
    s2 = LoCoMoScenario(
        id="locomo_02_auth_evolution",
        title="Authentication Protocol Evolution",
        sessions=[
            # Session 1
            [
                {"role": "user", "content": "We use JWT authentication with symmetric HS256 secret tokens stored in environment variables."},
                {"role": "assistant", "content": "Recorded: JWT with HS256 symmetric signing."},
            ],
            # Session 2
            [
                {"role": "user", "content": "Security audit complete. Update constraint: symmetric HS256 is deprecated due to key distribution risks. We must migrate to RS256 asymmetric keys."},
                {"role": "assistant", "content": "Security constraint updated: HS256 deprecated, migrate to RS256 asymmetric keys."},
            ],
            # Session 3
            [
                {"role": "user", "content": "Final rollout: All authentication now enforces RS256 with rotation via JWKS. Constraint: do NOT accept HS256 tokens under any circumstances."},
                {"role": "assistant", "content": "Confirmed. HS256 is forbidden. JWKS RS256 only."},
            ],
        ],
        probes=[
            ProbeQuestion(
                id="p4_auth_algo",
                question="What algorithm is required for signing authentication tokens?",
                expected="RS256 asymmetric keys via JWKS",
                accept=[r"\bRS256\b"],
                stale=[r"\bHS256\b"],
                is_deprecation=True,
            ),
            ProbeQuestion(
                id="p5_hs256_rejection",
                question="Is an HS256 signed token acceptable for internal APIs?",
                expected="No, HS256 is forbidden/deprecated.",
                kind="negative",
                # "not accept(ed)" is the fixture's own wording ("do NOT accept HS256
                # tokens"); the verb only, so a bare "not acceptable" stays ungrounded.
                accept=[r"\bRS256\b", r"\bforbidden\b", r"\bdeprecat\w*", r"\bprohibited\b", r"\breject\w*", r"\bsymmetric\b",
                        r"\bnot\s+accept(?:ed)?\b"],
                is_deprecation=True,
            ),
        ],
    )
    scenarios.append(s2)

    # Scenario 3: Caching & Rate Limiting Strategy
    s3 = LoCoMoScenario(
        id="locomo_03_cache_strategy",
        title="Caching & Rate Limiting Strategy",
        sessions=[
            # Session 1
            [
                {"role": "user", "content": "Cache layer is Memcached cluster on port 11211. TTL standard is 300 seconds."},
                {"role": "assistant", "content": "Understood: Memcached on 11211 with 300s TTL."},
            ],
            # Session 2
            [
                {"role": "user", "content": "Architecture change: Replace Memcached with Redis cluster on port 6379 for distributed locking and pub/sub. Memcached is retired."},
                {"role": "assistant", "content": "Memcached retired. Redis on port 6379 is now the active cache and lock store."},
            ],
        ],
        probes=[
            ProbeQuestion(
                id="p6_cache_engine",
                question="What system do we use for caching and distributed locks?",
                expected="Redis cluster on port 6379",
                accept=[r"\bRedis\b"],
                stale=[r"\bMemcached\b"],
                is_deprecation=True,
            ),
        ],
    )
    scenarios.append(s3)

    if smoke:
        return scenarios

    # Add 22 more scenarios for full 25-scenario evaluation.
    # (domain, v1, v2, v3, v1_re, v2_re, v3_re): the ingested text uses the
    # literal values; the regexes are what the judge accepts for each value,
    # so "Kubernetes." counts for "Kubernetes k8s" (scorer v1 required the
    # literal gold string).
    domain_topics = [
        ("api_format", "XML RPC", "JSON REST", "GraphQL",
         r"\bXML[\s-]*RPC\b", r"\bREST\b", r"\bGraphQL\b"),
        ("container_orchestration", "Docker Swarm", "Nomad", "Kubernetes k8s",
         r"\bSwarm\b", r"\bNomad\b", r"\bKubernetes\b|\bk8s\b"),
        ("logging_backend", "Local Syslog", "Logstash", "Vector to OpenSearch",
         r"\bSyslog\b", r"\bLogstash\b", r"\bOpenSearch\b"),
        ("python_version", "Python 3.9", "Python 3.11", "Python 3.12",
         r"\b3\.9\b", r"\b3\.11\b", r"\b3\.12\b"),
        ("ci_provider", "Jenkins", "Travis CI", "GitHub Actions",
         r"\bJenkins\b", r"\bTravis\b", r"\bGitHub\s+Actions\b"),
        ("message_broker", "RabbitMQ", "ActiveMQ", "Apache Kafka",
         r"\bRabbitMQ\b", r"\bActiveMQ\b", r"\bKafka\b"),
        ("rpc_framework", "Thrift", "REST", "gRPC with Protobuf v3",
         r"\bThrift\b", r"\bREST\b", r"\bgRPC\b"),
        ("cloud_region", "us-east-1", "us-west-2", "eu-central-1",
         r"\bus-east-1\b", r"\bus-west-2\b", r"\beu-central-1\b"),
        ("monitoring", "Nagios", "Graphite", "Prometheus with Grafana",
         r"\bNagios\b", r"\bGraphite\b", r"\bPrometheus\b"),
        ("code_formatter", "autopep8", "yapf", "Ruff and Black",
         r"\bautopep8\b", r"\byapf\b", r"\bRuff\b"),
        ("storage_tier", "S3 Standard", "S3 Glacier", "Ceph Object Storage",
         r"\bS3\s+Standard\b", r"\bGlacier\b", r"\bCeph\b"),
        ("tls_version", "TLS 1.1", "TLS 1.2", "Strict TLS 1.3",
         r"\bTLS\s*v?1\.1\b", r"\bTLS\s*v?1\.2\b", r"\bTLS\s*v?1\.3\b"),
        ("css_framework", "Bootstrap 4", "Bulma", "Tailwind CSS v3",
         r"\bBootstrap\b", r"\bBulma\b", r"\bTailwind\b"),
        ("build_system", "Makefiles", "Bazel", "Hatch / UV",
         r"\bMakefiles?\b", r"\bBazel\b", r"\bHatch\b|\buv\b"),
        ("dns_provider", "BIND9", "Route53", "Cloudflare DNS",
         r"\bBIND\s*9\b", r"\bRoute\s*53\b", r"\bCloudflare\b"),
        ("search_engine", "Solr", "Elasticsearch 6", "Kuzu Hybrid Search",
         r"\bSolr\b", r"\bElasticsearch\b", r"\bKuzu\b"),
        ("serialization", "Pickle", "MessagePack", "Protobuf",
         r"\bPickle\b", r"\bMessagePack\b|\bmsgpack\b", r"\bProtobuf\b|\bProtocol\s+Buffers\b"),
        ("tracing", "Jaeger standalone", "Zipkin", "OpenTelemetry OTel",
         r"\bJaeger\b", r"\bZipkin\b", r"\bOpenTelemetry\b|\bOTel\b"),
        ("testing_framework", "unittest", "nose2", "pytest",
         r"\bunittest\b", r"\bnose2?\b", r"\bpytest\b"),
        ("frontend_framework", "AngularJS", "Vue 2", "React 19 with Next.js",
         r"\bAngular(?:JS)?\b", r"\bVue\b", r"\bReact\b|\bNext\.?js\b"),
        ("secrets_manager", "Vault dev", "AWS Secrets Manager", "Infisical",
         r"\bVault\b", r"\bSecrets\s+Manager\b", r"\bInfisical\b"),
        ("os_base_image", "Ubuntu 18.04", "CentOS 7", "Debian 12 Bookworm slim",
         r"\bUbuntu\b", r"\bCentOS\b", r"\bDebian\b"),
    ]

    for idx, (domain, v1, v2, v3, re1, re2, re3) in enumerate(domain_topics, start=4):
        sc = LoCoMoScenario(
            id=f"locomo_{idx:02d}_{domain}",
            title=f"Evolution of {domain}",
            sessions=[
                [
                    {"role": "user", "content": f"Constraint for {domain}: we use {v1}."},
                    {"role": "assistant", "content": f"Acknowledged {v1} for {domain}."},
                ],
                [
                    {"role": "user", "content": f"Update: {v1} is deprecated; migrate {domain} to {v2}."},
                    {"role": "assistant", "content": f"Updated {domain} to {v2}."},
                ],
                [
                    {"role": "user", "content": f"Final decision: {v2} has been replaced by {v3}. Constraint: strictly use {v3} for {domain}, do NOT use {v1} or {v2}."},
                    {"role": "assistant", "content": f"Confirmed {v3} is now standard for {domain}."},
                ],
            ],
            probes=[
                ProbeQuestion(
                    id=f"p_{idx}_active",
                    question=f"What is the current required tool for {domain}?",
                    expected=v3,
                    accept=[re3],
                    stale=[re1, re2],
                    is_deprecation=True,
                ),
            ],
        )
        scenarios.append(sc)

    return scenarios
