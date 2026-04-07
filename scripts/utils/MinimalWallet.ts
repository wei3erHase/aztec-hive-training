import {
  type AccountWithSecretKey,
  type Account,
  type NoFrom,
  NO_FROM,
} from '@aztec/aztec.js/account';
import { AztecAddress } from '@aztec/aztec.js/addresses';
import type { AztecNode } from '@aztec/aztec.js/node';
import { DefaultMultiCallEntrypoint } from '@aztec/entrypoints/multicall';
import type { PXE } from '@aztec/pxe/server';
import { BaseWallet, type FeeOptions } from '@aztec/wallet-sdk/base-wallet';
import type { ExecutionPayload, TxExecutionRequest } from '@aztec/stdlib/tx';

export class MinimalWallet extends BaseWallet {
  private readonly addressToAccount = new Map<string, AccountWithSecretKey>();

  constructor(pxe: PXE, aztecNode: AztecNode) {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    super(pxe as unknown as any, aztecNode);
  }

  /**
   * Add an account to this wallet's internal registry
   */
  public addAccount(account: AccountWithSecretKey): void {
    this.addressToAccount.set(account.getAddress().toString(), account);
  }

  /**
   * Get an account by its address
   */
  protected async getAccountFromAddress(
    address: AztecAddress
  ): Promise<Account> {
    const account = this.addressToAccount.get(address.toString());
    if (!account) {
      throw new Error(
        `Account not found in wallet for address: ${address.toString()}`
      );
    }
    return account;
  }

  /**
   * Override to use DefaultMultiCallEntrypoint for NO_FROM transactions.
   *
   * The base class uses DefaultEntrypoint for NO_FROM which only accepts a
   * single call. Contract deployments involve multiple calls (class publication,
   * instance registration, constructor), so we need DefaultMultiCallEntrypoint.
   */
  protected override async createTxExecutionRequestFromPayloadAndFee(
    executionPayload: ExecutionPayload,
    from: AztecAddress | NoFrom,
    feeOptions: FeeOptions
  ): Promise<TxExecutionRequest> {
    if (from === NO_FROM) {
      const feeExecutionPayload =
        await feeOptions.walletFeePaymentMethod?.getExecutionPayload();
      const { mergeExecutionPayloads } = await import('@aztec/stdlib/tx');
      const finalPayload = feeExecutionPayload
        ? mergeExecutionPayloads([feeExecutionPayload, executionPayload])
        : executionPayload;
      const chainInfo = await this.getChainInfo();
      // Cast through any: @aztec/entrypoints bundles its own nested copy of
      // @aztec/stdlib, so TypeScript sees structurally-identical types as
      // incompatible (separate declarations of private 'xCoord'). Safe at runtime.
      const entrypoint = new DefaultMultiCallEntrypoint();
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      return (entrypoint as any).createTxExecutionRequest(
        finalPayload,
        feeOptions.gasSettings,
        chainInfo
      ) as TxExecutionRequest;
    }
    return super.createTxExecutionRequestFromPayloadAndFee(
      executionPayload,
      from,
      feeOptions
    );
  }

  /**
   * Get all accounts registered with this wallet
   */
  async getAccounts(): Promise<{ alias: string; item: AztecAddress }[]> {
    return Array.from(this.addressToAccount.values()).map((acc) => ({
      alias: '',
      item: acc.getAddress(),
    }));
  }
}
