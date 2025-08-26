package com.ssafy.ssashinsa.heyfy.register.service;

import com.ssafy.ssashinsa.heyfy.account.domain.Account;
import com.ssafy.ssashinsa.heyfy.account.domain.ForeignAccount;
import com.ssafy.ssashinsa.heyfy.account.repository.AccountRepository;
import com.ssafy.ssashinsa.heyfy.account.repository.ForeignAccountRepository;
import com.ssafy.ssashinsa.heyfy.common.exception.CustomException;
import com.ssafy.ssashinsa.heyfy.common.util.SecurityUtil;
import com.ssafy.ssashinsa.heyfy.register.exception.ShinhanRegisterApiErrorCode;
import com.ssafy.ssashinsa.heyfy.shinhanApi.client.ShinhanDemandDepositApiClient;
import com.ssafy.ssashinsa.heyfy.shinhanApi.client.ShinhanForeignDemandDepositApiClient;
import com.ssafy.ssashinsa.heyfy.shinhanApi.dto.account.create.ShinhanCreateDepositResponseDto;
import com.ssafy.ssashinsa.heyfy.shinhanApi.dto.account.inquire.ShinhanInquireSingleDepositResponseDto;
import com.ssafy.ssashinsa.heyfy.user.domain.Users;
import com.ssafy.ssashinsa.heyfy.user.repository.UserRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

@Service
@Slf4j
@RequiredArgsConstructor
public class RegisterService {

    private final UserRepository userRepository;
    private final AccountRepository accountRepository;
    private final ForeignAccountRepository foreignAccountRepository;
    private final ShinhanDemandDepositApiClient shinhanDemandDepositApiClient;
    private final ShinhanForeignDemandDepositApiClient shinhanForeignDemandDepositApiClient;


    public ShinhanCreateDepositResponseDto createDepositAccount() {
        String studentId = SecurityUtil.getCurrentStudentId();
        Users user = userRepository.findByStudentId(studentId)
                .orElseThrow(() -> new CustomException(ShinhanRegisterApiErrorCode.USER_NOT_FOUND));

        String userKey = user.getUserKey();
        if (userKey == null || userKey.isEmpty()) {
            throw new CustomException(ShinhanRegisterApiErrorCode.MISSING_USER_KEY);
        }

        if (accountRepository.findByUser(user).isPresent()) {
            log.warn("이미 계좌가 존재하는 유저입니다. studentId: {}", studentId);
            throw new CustomException(ShinhanRegisterApiErrorCode.ACCOUNT_ALREADY_EXISTS);
        }

        ShinhanCreateDepositResponseDto response = shinhanDemandDepositApiClient.createDemandDepositAccount(userKey);
        String accountNo = response.getREC().getAccountNo();

        Account account = Account.builder()
                .user(user)
                .accountNo(accountNo)
                .build();
        accountRepository.save(account);

        return response;
    }

    public ShinhanCreateDepositResponseDto createForeignDepositAccount() {
        String studentId = SecurityUtil.getCurrentStudentId();
        Users user = userRepository.findByStudentId(studentId)
                .orElseThrow(() -> new CustomException(ShinhanRegisterApiErrorCode.USER_NOT_FOUND));

        String userKey = user.getUserKey();
        if (userKey == null || userKey.isEmpty()) {
            throw new CustomException(ShinhanRegisterApiErrorCode.MISSING_USER_KEY);
        }

        if (foreignAccountRepository.findByUser(user).isPresent()) {
            log.warn("이미 계좌가 존재하는 유저입니다. studentId: {}", studentId);
            throw new CustomException(ShinhanRegisterApiErrorCode.ACCOUNT_ALREADY_EXISTS);
        }

        ShinhanCreateDepositResponseDto response = shinhanForeignDemandDepositApiClient.createForeignCurrencyDemandDepositAccount(userKey);

        String accountNo = response.getREC().getAccountNo();
        String currencyCode = response.getREC().getCurrency().getCurrency();

        ForeignAccount foreignAccount = ForeignAccount.builder()
                .user(user)
                .accountNo(accountNo)
                .currency(currencyCode)
                .build();
        foreignAccountRepository.save(foreignAccount);

        return response;
    }

    @Transactional
    public String registerAccountFromShinhan(String accountNo) {
        String studentId = SecurityUtil.getCurrentStudentId();
        log.info("학생 ID [{}]의 계좌 등록을 시작합니다.", studentId);

        Users user = userRepository.findByStudentId(studentId)
                .orElseThrow(() -> new CustomException(ShinhanRegisterApiErrorCode.USER_NOT_FOUND));

        String userKey = user.getUserKey();

        ShinhanInquireSingleDepositResponseDto response = shinhanDemandDepositApiClient.inquireDemandDepositAccount(userKey, accountNo);
        registerAccount(accountNo);

        return accountNo;
    }

    @Transactional
    public String registerForeignAccountFromShinhan(String accountNo) {
        String studentId = SecurityUtil.getCurrentStudentId();
        log.info("학생 ID [{}]의 외화 계좌 등록을 시작합니다.", studentId);

        Users user = userRepository.findByStudentId(studentId)
                .orElseThrow(() -> new CustomException(ShinhanRegisterApiErrorCode.USER_NOT_FOUND));

        String userKey = user.getUserKey();

        ShinhanInquireSingleDepositResponseDto response = shinhanForeignDemandDepositApiClient.inquireForeignCurrencyDemandDepositAccount(userKey, accountNo);
        registerForeignAccount(accountNo, response.getREC().getCurrency());

        return accountNo;
    }


    @Transactional
    public void registerAccount(String accountNo) {
        String studentId = SecurityUtil.getCurrentStudentId();
        log.info("학생 ID [{}]의 계좌 등록을 시작합니다.", studentId);

        Users user = userRepository.findByStudentId(studentId)
                .orElseThrow(() -> new CustomException(ShinhanRegisterApiErrorCode.USER_NOT_FOUND));

        if (accountRepository.existsByUser(user)) {
            user.setAccount(null);
            accountRepository.deleteByUser(user);
            user = userRepository.getReferenceById(user.getId());
        }

        Account newAccount = Account.builder()
                .user(user)
                .accountNo(accountNo)
                .build();

        accountRepository.save(newAccount);

        log.info("학생 ID [{}]의 일반 계좌가 성공적으로 개설되었습니다. 계좌번호: {}", user.getStudentId(), accountNo);
    }

    @Transactional
    public void registerForeignAccount(String accountNo, String currency) {
        String studentId = SecurityUtil.getCurrentStudentId();
        log.info("학생 ID [{}]의 외화 계좌 등록을 시작합니다.", studentId);

        Users user = userRepository.findByStudentId(studentId)
                .orElseThrow(() -> new CustomException(ShinhanRegisterApiErrorCode.USER_NOT_FOUND));

        if (foreignAccountRepository.existsByUser(user)) {
            user.setAccount(null);
            foreignAccountRepository.deleteByUser(user);
            user = userRepository.getReferenceById(user.getId());
        }

        ForeignAccount newAccount = ForeignAccount.builder()
                .user(user)
                .accountNo(accountNo)
                .currency(currency)
                .build();

        foreignAccountRepository.save(newAccount);

        log.info("학생 ID [{}]의 외환 계좌가 성공적으로 개설되었습니다. 계좌번호: {}", user.getStudentId(), accountNo);
    }

    // 외부에서 직접 호출되지 않도록 private 메서드로 변경
    private void createForeignDepositAccount(Users user) {
        try {
            String userKey = user.getUserKey();
            if (userKey == null || userKey.isEmpty()) {
                throw new CustomException(ShinhanRegisterApiErrorCode.MISSING_USER_KEY);
            }

            ShinhanCreateDepositResponseDto response = shinhanForeignDemandDepositApiClient.createForeignCurrencyDemandDepositAccount(userKey);

            String accountNo = response.getREC().getAccountNo();
            String currencyCode = response.getREC().getCurrency().getCurrency();

            ForeignAccount foreignAccount = ForeignAccount.builder()
                    .user(user)
                    .accountNo(accountNo)
                    .currency(currencyCode)
                    .build();
            foreignAccountRepository.save(foreignAccount);

            log.info("학생 ID [{}]의 외화 계좌가 성공적으로 개설되었습니다. 계좌번호: {}", user.getStudentId(), accountNo);

        } catch (CustomException ce) {
            log.error("학생 ID [{}]의 외화 계좌 개설 중 커스텀 예외 발생: {}", user.getStudentId(), ce.getMessage());
        } catch (Exception e) {
            log.error("학생 ID [{}]의 외화 계좌 개설 API 호출 실패: {}", user.getStudentId(), e.getMessage(), e);
        }
    }


}